"""Same identity-only selection as the already frozen original 800 parents."""
import hashlib

NAMESPACE='runtime-necessity-minimal-v2-20260925'
def selected_keys():
    def h(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
    parents=sorted((f'eval-{p:04d}' for p in range(1600)),key=lambda p:h(f'{NAMESPACE}|parent|{p}'))[:800]
    return sorted((p,min(range(3),key=lambda r:h(f'{NAMESPACE}|repeat-{r}|{p}'))) for p in parents)
