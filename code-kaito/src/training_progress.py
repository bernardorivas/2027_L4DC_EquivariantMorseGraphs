"""Live notebook loss curves from the worker's existing flushed JSON log."""
from pathlib import Path
import json,subprocess,time

def read_history(run):
    path=Path(run)/'training.log';rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except (ValueError,TypeError):continue
            if all(k in row for k in ('stage','update','train','validation')):rows.append(row)
    return rows

class LossPlot:
    def __init__(self):
        import matplotlib.pyplot as plt
        from IPython import get_ipython
        from IPython.display import display
        self.plt=plt;self.handle=None;self.last=None
        with plt.rc_context({'font.family':'serif','mathtext.fontset':'cm','font.size':11}):
            self.fig,self.ax=plt.subplots(figsize=(6.3,3.1),layout='constrained')
        plt.close(self.fig)
        self.display=display if get_ipython() is not None else None

    def update(self,rows):
        if not rows or rows[-1]==self.last:return
        self.last=rows[-1];stage=rows[-1]['stage'];rows=[r for r in rows if r['stage']==stage]
        self.ax.clear();steps=[r['update'] for r in rows]
        self.ax.plot(steps,[r['train'] for r in rows],label='Training (batch)',color='#0072B2')
        self.ax.plot(steps,[r['validation'] for r in rows],label='Validation',color='#D55E00')
        self.ax.set(xlabel='Update',ylabel='Loss',yscale='log',title=stage.replace('_',' ').capitalize()+' loss')
        self.ax.legend(frameon=False);self.ax.grid(axis='y',alpha=.18)
        if self.display:
            if self.handle is None:self.handle=self.display(self.fig,display_id=True)
            else:self.handle.update(self.fig)

    def save(self,run):
        if self.last:
            for ext in ('png','pdf'):self.fig.savefig(Path(run)/f'loss-curve.{ext}',dpi=170,bbox_inches='tight')

def show_loss(run):
    curve=LossPlot();curve.update(read_history(run));curve.save(run)

def run_with_progress(command,run,env,live=True,timeout=900):
    run=Path(run);curve=LossPlot() if live else None;start=time.monotonic()
    with (run/'training.log').open('w') as log:
        with subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT) as process:
            try:
                while process.poll() is None:
                    if time.monotonic()-start>timeout:raise subprocess.TimeoutExpired(command,timeout)
                    if curve:curve.update(read_history(run))
                    time.sleep(.3)
            except BaseException:
                process.kill();process.wait();raise
    if curve:curve.update(read_history(run));curve.save(run)
    return process.returncode
