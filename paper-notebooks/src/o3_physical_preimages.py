"""Sample the actual encoder on orbit representatives (a,r,0,0)."""
import sys
from pathlib import Path
import numpy as np
from quotient_runtime import load_quotient
fit, output = map(Path,sys.argv[1:])
_, encode, _, _ = load_quotient(fit)
a=np.linspace(-1.5,1.5,501);r=np.linspace(0,1.5,251)
aa,rr=np.meshgrid(a,r)
x=np.column_stack((aa.ravel(),rr.ravel(),np.zeros(aa.size),np.zeros(aa.size)))
q=np.concatenate([np.asarray(encode(x[i:i+4096])) for i in range(0,len(x),4096)])
# Check that a second representative of each orbit receives the same coordinates.
test=x[::503].copy();test[:,[1,2]]=test[:,[2,1]]
err=float(np.max(np.abs(np.asarray(encode(test))-q[::503])))
assert err<1e-8
np.savez_compressed(output,a=a,r=r,encoded=q.reshape(len(r),len(a),2),orbit_error=err)
