#!/usr/bin/env python3
import argparse
from espnet.nets.batch_beam_search_online import BatchBeamSearchOnline
from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,  # noqa: H301
)
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
import logging
import numpy as np
from pathlib import Path
import sys
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union


class Speech2TextStreaming:

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        penalty: float = 0.0,
        nbest: int = 1,
        disable_repetition_detection=False,
        decoder_text_length_limit=0,
        encoded_feat_length_limit=0,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        assert isinstance(
            asr_model.encoder, ContextualBlockTransformerEncoder
        ) or isinstance(asr_model.encoder, ContextualBlockConformerEncoder)

        decoder = asr_model.decoder
        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build BeamSearch object
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            length_bonus=penalty,
        )

        assert "encoder_conf" in asr_train_args
        assert "look_ahead" in asr_train_args.encoder_conf
        assert "hop_size" in asr_train_args.encoder_conf
        assert "block_size" in asr_train_args.encoder_conf

        assert batch_size == 1

        beam_search = BatchBeamSearchOnline(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
            disable_repetition_detection=disable_repetition_detection,
            decoder_text_length_limit=decoder_text_length_limit,
            encoded_feat_length_limit=encoded_feat_length_limit,
        )

        non_batch = [
            k
            for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        assert len(non_batch) == 0

        logging.info("BatchBeamSearchOnline implementation is selected.")

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

        self.reset()

    def reset(self):
        self.frontend_states = None
        self.encoder_states = None
        self.beam_search.reset()


    def apply_frontend(
        self, speech: torch.Tensor, prev_states=None, is_final: bool = False
    ):
        if prev_states is not None:
            buf = prev_states["waveform_buffer"]
            speech = torch.cat([buf, speech], dim=0)

        if is_final:
            speech_to_process = speech
            waveform_buffer = None
        else:
            n_frames = (speech.size(0) - 384) // 128
            n_residual = (speech.size(0) - 384) % 128
            speech_to_process = speech.narrow(0, 0, 384 + n_frames * 128)
            waveform_buffer = speech.narrow(
                0, speech.size(0) - 384 - n_residual, 384 + n_residual
            ).clone()

        # data: (Nsamples,) -> (1, Nsamples)
        speech_to_process = speech_to_process.unsqueeze(0).to(
            getattr(torch, self.dtype)
        )
        lengths = speech_to_process.new_full(
            [1], dtype=torch.long, fill_value=speech_to_process.size(1)
        )
        batch = {"speech": speech_to_process, "speech_lengths": lengths}

        # lenghts: (1,)
        # a. To device
        batch = to_device(batch, device=self.device)

        feats, feats_lengths = self.asr_model._extract_feats(**batch)
        if self.asr_model.normalize is not None:
            feats, feats_lengths = self.asr_model.normalize(feats, feats_lengths)

        # Trimming
        if is_final:
            if prev_states is None:
                pass
            else:
                feats = feats.narrow(1, 2, feats.size(1) - 2)
        else:
            if prev_states is None:
                feats = feats.narrow(1, 0, feats.size(1) - 2)
            else:
                feats = feats.narrow(1, 2, feats.size(1) - 4)

        feats_lengths = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        if is_final:
            next_states = None
        else:
            next_states = {"waveform_buffer": waveform_buffer}
        return feats, feats_lengths, next_states


    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray], is_final: bool = True
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        feats, feats_lengths, self.frontend_states = self.apply_frontend(
            speech, self.frontend_states, is_final=is_final
        )
        enc, _, self.encoder_states = self.asr_model.encoder(
            feats,
            feats_lengths,
            self.encoder_states,
            is_final=is_final,
            infer_mode=True,
        )
        nbest_hyps = self.beam_search(
            x=enc[0],
            maxlenratio=self.maxlenratio,
            minlenratio=self.minlenratio,
            is_final=is_final,
        )

        ret = self.assemble_hyps(nbest_hyps)
        if is_final:
            self.reset()
        return ret

    def assemble_hyps(self, hyps):
        nbest_hyps = hyps[: self.nbest]
        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results