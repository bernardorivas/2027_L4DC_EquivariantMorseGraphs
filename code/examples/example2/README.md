# Example 2: O(3) flow

This is the physical system in Section 5.2. Run commands from `code/` after the [shared setup](../../README.md#setup).

```bash
python examples/example2/run.py replay --out outputs/example2-replay
```

Replay checks the supplied input checksums, recomputes the explicit image enclosures, runs native CMGDB on the explicit and saved fitted images, verifies cells, edges, quotient annotations and grouped regions, and writes the six manuscript panels under the output's `figures/` directory. The saved fitted model uses the earlier observation-based workflow. A new joint-training command and checkpoint are not included in this folder.

## Physical system

For $(a,v)\in[-3/2,3/2]\times B^3_{3/2}$,

$$\dot a=a(1-a^2),\qquad \dot v=(1-2a^2-\|v\|^2)v.$$

The action is $(a,v)\mapsto(a,Qv)$ for $Q\in O(3)$. The quotient coordinates are $(a,r)=(a,\|v\|)$. At time $\log3$ the exact quotient map is

$$F(a,r)=\left(\frac{3a}{\sqrt D},\frac{3r}{\sqrt{D(D+8r^2)}}\right),\qquad D=1+8a^2.$$

The nonempty invariant-set order is $O\to C\to A_\pm$. The supplied $2^7$ by $2^7$ explicit grid has five native components. The display groups the extra component, whose invariant set is empty, with the saddle. The fitted grid has four native components. Raw and grouped outputs are both retained. Native quotient annotations and the analytic original-space index labels have separate sources.

## Generate data

```bash
python examples/example2/run.py data --count 16384 --seed 1 --intervals 10 --out outputs/example2-data
```

This generates independent physical starts and their closed-form trajectories at spacing $\log3$. `trajectories.npz` contains `states` with shape `(count, intervals + 1, 4)`, ordered `(a,v1,v2,v3)`, and `times`. The accompanying JSON records the generation settings. Generate validation and test trajectories with different seeds and split by trajectory before forming pairs. These newly generated data are separate from replaying the saved manuscript result.
