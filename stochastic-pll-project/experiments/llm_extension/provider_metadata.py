"""Read-only Azure metadata checks; never creates a model response."""
from pathlib import Path
import urllib.request,urllib.error,json,datetime
BASE='https://tiwarishrayansh-0806-resource.services.ai.azure.com'
KEY=Path('/tmp/stochastic-pll-azure-key').read_text().strip()
checks=[]
urls=[BASE+'/openai/v1/models',BASE+'/api/projects/tiwarishrayansh-0806/deployments/gpt-5-mini?api-version=v1']
for url in urls:
 req=urllib.request.Request(url,headers={'api-key':KEY,'Accept':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=30) as r:
   body=json.load(r);status=r.status;headers={k:v for k,v in r.headers.items() if k.lower().startswith(('x-ratelimit','retry-after','x-ms-region'))}
  checks.append({'url':url,'http_status':status,'body':body,'rate_headers':headers})
 except urllib.error.HTTPError as e:
  raw=e.read().decode(errors='replace').replace(KEY,'[REDACTED]');checks.append({'url':url,'http_status':e.code,'error_body':raw[:1000]})
 except Exception as e:checks.append({'url':url,'error_type':type(e).__name__})
result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inference_calls':0,'checks':checks}
p=Path('experiments/llm_extension/pilot/provider_metadata.json');p.write_text(json.dumps(result,indent=2))
for c in checks:
 b=c.get('body',{});items=b.get('data',[])
 print(json.dumps({'http_status':c.get('http_status'),'error_type':c.get('error_type'),'models':[x.get('id') for x in items if 'gpt-5-mini' in str(x.get('id'))],'deployment_metadata':b if 'data' not in b else None,'rate_headers':c.get('rate_headers')}))
