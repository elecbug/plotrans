"""Reproduce the isolated RapidLaTeXOCR comparison on the supplied PDF fixtures."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
import urllib.request


CASES = [
    ("pastry-b", "Pastry", "6e592f20df71", 16, [302.8036,261.6500,323.7636,270.0100], "b=4"),
    ("pastry-leaf", "Pastry", "6e592f20df71", 16, [328.8052,259.4900,360.9252,270.4900], "|L|=16"),
    ("pastry-neighbors", "Pastry", "6e592f20df71", 16, [366.0868,259.4900,401.8068,270.4900], "|M|=32"),
    ("bitcoin-piecewise", "Bitcoin", "b1674191a88e", 6, [125.0,140.142,226.799,178.918], "q_z=cases(1 if p<=q; (q/p)^z if p>q)"),
    ("bitcoin-poisson", "Bitcoin", "b1674191a88e", 7, [125.5698,473.5570,155.7830,496.2620], "lambda=z*q/p"),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--models",type=Path,default=Path(".tmp/math-ocr-models"))
    parser.add_argument("--output",type=Path,default=Path("outputs/math-comparison.json"))
    parser.add_argument("--download-models",action="store_true")
    parser.add_argument("--scale",type=float,default=2)
    args=parser.parse_args()
    manifest=json.loads(Path(__file__).with_name("math-ocr-models.json").read_text())
    args.models.mkdir(parents=True,exist_ok=True)
    for filename,expected in manifest.items():
        path=args.models/filename
        if not path.exists() and args.download_models:
            url="https://github.com/RapidAI/RapidLaTeXOCR/releases/download/v0.0.0/"+filename
            temporary=path.with_suffix(".part")
            print("Downloading",filename,flush=True)
            with urllib.request.urlopen(url,timeout=60) as response,temporary.open("wb") as output:
                while data:=response.read(1024*1024):output.write(data)
            with temporary.open("rb") as file:actual=hashlib.file_digest(file,"sha256").hexdigest()
            if actual!=expected:raise ValueError("Model checksum mismatch: "+filename)
            temporary.replace(path)
        if not path.exists():raise FileNotFoundError("Prepare models with --download-models: "+str(path))
        with path.open("rb") as file:actual=hashlib.file_digest(file,"sha256").hexdigest()
        if actual!=expected:raise ValueError("Model checksum mismatch: "+filename)

    import pypdfium2 as pdfium
    from rapid_latex_ocr import LaTeXOCR,main as main_module,models as model_module
    from rapid_latex_ocr.utils_load import OrtInferSession
    class Session(OrtInferSession):
        def __init__(self,model_path,num_threads=4):super().__init__(model_path,num_threads=4)
    main_module.OrtInferSession=Session;model_module.OrtInferSession=Session
    engine=LaTeXOCR(image_resizer_path=args.models/"image_resizer.onnx",encoder_path=args.models/"encoder.onnx",
                   decoder_path=args.models/"decoder.onnx",tokenizer_json=args.models/"tokenizer.json")
    results=[]
    for label,fragment,digest,page_number,box,expected in CASES:
        source=next(p for p in args.input.glob("*.pdf") if fragment.lower() in p.name.lower())
        with source.open("rb") as file:actual=hashlib.file_digest(file,"sha256").hexdigest()
        if not actual.startswith(digest):raise ValueError("Fixture differs from reviewed PDF: "+source.name)
        with pdfium.PdfDocument(source) as pdf:
            page=pdf[page_number-1];w,h=page.get_size();l,b,r,t=box
            bitmap=page.render(scale=args.scale,crop=(l,b,w-r,h-t));image=bitmap.to_pil()
            try:
                buffer=BytesIO();image.save(buffer,format="PNG")
                prediction,seconds=engine(buffer.getvalue())
            finally:image.close();bitmap.close();page.close()
        row={"case":label,"source":source.name,"page":page_number,"expected":expected,"actual":prediction,"seconds":seconds}
        results.append(row);print(json.dumps(row,ensure_ascii=True),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"library":"rapid-latex-ocr==0.0.9","scale":args.scale,
        "model_sha256":manifest,"cases":results},indent=2),encoding="utf-8")


if __name__=="__main__":main()
