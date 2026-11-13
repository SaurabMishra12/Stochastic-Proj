"""Download two public author-distributed datasets without using model APIs."""
from pathlib import Path
import urllib.request,json,hashlib
root=Path(__file__).resolve().parent/'source_data';root.mkdir(exist_ok=True)
urls={'hotpotqa_dev_distractor.json':'https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json','bbh_logical_deduction_three_objects.json':'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/logical_deduction_three_objects.json'}
manifest={}
for name,url in urls.items():
 p=root/name
 if not p.exists():
  req=urllib.request.Request(url,headers={'User-Agent':'StochasticPLLResearch/1.0 public-dataset-download'})
  data=urllib.request.urlopen(req,timeout=90).read();json.loads(data);p.write_bytes(data)
 data=p.read_bytes();raw=json.loads(data);count=len(raw['examples']) if isinstance(raw,dict) else len(raw)
 manifest[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'records':count}
 print(name,count,len(data))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
