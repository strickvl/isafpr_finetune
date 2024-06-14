from rich import print
import json
import os
import re

import modal
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from modal import Image

datascience_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("peft", "transformers")
    .env({"HF_TOKEN": os.getenv("HF_TOKEN")})
)

app = modal.App("inference-llama3")


@app.function(image=datascience_image, gpu="a100")
def run_inference():
    model_id = "strickvl/isafpr-llama3-lora"
    model = AutoPeftModelForCausalLM.from_pretrained(model_id).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def prompt(press_release):
        return f"""You are an expert at identifying events in a press release. You are precise and always make sure you are correct, drawing inference from the text of the press release. event_types = ['airstrike', 'detention', 'captureandkill', 'insurgentskilled', 'exchangeoffire', 'civiliancasualty'], provinces = ['badakhshan', 'badghis', 'baghlan', 'balkh', 'bamyan', 'day_kundi', 'farah', 'faryab', 'ghazni', 'ghor', 'helmand', 'herat', 'jowzjan', 'kabul', 'kandahar', 'kapisa', 'khost', 'kunar', 'kunduz', 'laghman', 'logar', 'nangarhar', 'nimroz', 'nuristan', 'paktya', 'paktika', 'panjshir', 'parwan', 'samangan', 'sar_e_pul', 'takhar', 'uruzgan', 'wardak', 'zabul'], target_groups = ['taliban', 'haqqani', 'criminals', 'aq', 'hig', 'let', 'imu', 'judq', 'iju', 'hik', 'ttp', 'other']

    ### Instruction:

    PRESS RELEASE TEXT: "{press_release}"

    ### Response:
    """

    def prompt_tok(press_release, return_ids=False):
        _p = prompt(press_release)
        input_ids = tokenizer(_p, return_tensors="pt", truncation=True).input_ids.cuda()
        out_ids = model.generate(
            input_ids=input_ids, max_new_tokens=5000, do_sample=False
        )
        ids = out_ids.detach().cpu().numpy()
        if return_ids:
            return out_ids
        return tokenizer.batch_decode(ids, skip_special_tokens=True)[0][len(_p) :]

    pr1 = """2011-11-S-011 ISAF Joint Command - Afghanistan For Immediate Release KABUL, Afghanistan (Nov. 7, 2011) — A combined Afghan and coalition security force conducted an operation in search of a Haqqani facilitator in Argo district, Badakshan province. The facilitator coordinates suicide attacks with other insurgent leaders in the area. During the operation, a local national male failed to comply with repeated verbal warnings and displayed hostile intent toward the security force. The security force engaged the individual, resulting in his death. The security force confiscated a shotgun and intelligence linking the local national to the Haqqani network. The security force also detained two suspected insurgents during the operation."""

    out = prompt_tok(pr1).replace("</s>", "").strip()
    print(out)

    # Remove newline characters from the output string
    out = out.replace("\n", "")

    # Add missing double quotes around keys and values using regex
    out = re.sub(r"(\w+):", r'"\1":', out)
    out = re.sub(r":(\w+)", r':"\1"', out)
    print(out)
    out_dict = json.loads(out)
    return out_dict


@app.local_entrypoint()
def main():
    # run the function remotely on Modal
    print(run_inference.remote())
