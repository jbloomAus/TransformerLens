import gc

import einops
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device

device = get_device()
torch.set_grad_enabled(False)
dtype = torch.float32

ooms = [10**-i for i in range(1, 10)]


def assert_close_for_ooms(a, b, ooms=ooms):
    for oom in ooms:
        assert torch.allclose(a, b, rtol=oom, atol=oom), f"Failed for oom={oom}, max diff={torch.max(torch.abs(a - b))}"

def check_gemma_2b_hf_vs_tlens():

    torch.set_grad_enabled(False)
    model_name = "google/gemma-2b"

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )  # trust_remote_code=True, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )  # add_bos_token = True, use_fast=False, trust_remote_code=True)
    hf_model.eval().to(device)
    
    hooked_model = HookedTransformer.from_pretrained(
        model_name,
        tokenizer=tokenizer,
        fold_ln=False,
        fold_value_biases=False,
        center_writing_weights=False,
    )
    
    text = """
    TransformerLens lets you load in 50+ different open source language models,
    and exposes the internal activations of the model to you. You can cache
    any internal activation in the model, and add in functions to edit, remove
    or replace these activations as the model runs.
    """
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        hf_outputs = hf_model(input_ids, output_hidden_states=True, output_attentions=True)
        hf_logits_cpu = hf_outputs["logits"].cpu()
        hf_resid_pre_cache = hf_outputs["hidden_states"]
        hf_attentions = hf_outputs["attentions"]
        hf_resid_pre_cache_cpu = [cache.cpu() for cache in hf_resid_pre_cache]
        hf_attentions_cpu = [att.cpu() for att in hf_attentions]
        hf_outputs = hf_model(input_ids, labels=input_ids)
        hf_loss_cpu = hf_outputs.loss.cpu()

    # TODO: add a some notebook config for low memory mode.
    del hf_model
    del hf_outputs
    del hf_resid_pre_cache
    gc.collect()
    # torch.cuda.empty_cache()

    with torch.no_grad():
        hooked_model_logits, hooked_model_cache = hooked_model.run_with_cache(input_ids)
        hooked_model_loss = hooked_model(input_ids, return_type="loss")
        hooked_model_loss_cpu = hooked_model_loss.cpu()
        hooked_model_logits_cpu = hooked_model_logits.detach().cpu()
        hooked_model_cache_cpu = {k: v.cpu() for k, v in hooked_model_cache.items()}
        n_layers = hooked_model.cfg.n_layers

    # TODO: add a some notebook config for low memory mode.
    del hooked_model
    del hooked_model_logits
    del hooked_model_cache
    del hooked_model_loss

    gc.collect()
    # torch.cuda.empty_cache()

    
    
    pass_loose_bound = True
    print("*"*5, "Matching hf and T-Lens residual stream in between transformer blocks", "*"*5)
    atol = rtol = 1e-4
    print("*"*5, f"\ttesting with {atol=} and {rtol=}\t","*"*5)
    for l in range(n_layers):
        a = hooked_model_cache_cpu[f'blocks.{l}.hook_resid_pre']
        b = hf_resid_pre_cache_cpu[l]
        max_diff = (a - b).abs().max()
        try:
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        except:
            print(f"layer {l} \t not close, max difference: {max_diff}")
    
    assert_close_for_ooms(hf_logits_cpu, hooked_model_logits_cpu)
    
if __name__ == "__main__":
    check_gemma_2b_hf_vs_tlens()
    print("All tests passed")