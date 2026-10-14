"""Split a HepMC3 ascii file into N shards of K consecutive events (renumbered from 0).
python split_hepmc.py in.hepmc3 NEV_PER_SHARD  -> in_s00.hepmc3, in_s01.hepmc3, ..."""
import sys
inp, k = sys.argv[1], int(sys.argv[2])
head = "HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\n"
tail = "HepMC::Asciiv3-END_EVENT_LISTING\n"
f, iev, shard = None, -1, -1
for line in open(inp):
    if line.startswith("HepMC::"):
        continue
    if line.startswith("E "):
        iev += 1
        if iev % k == 0:
            if f: f.write(tail); f.close()
            shard += 1
            f = open(inp.replace(".hepmc3", f"_s{shard:02d}.hepmc3"), "w"); f.write(head)
        s = line.split(); s[1] = str(iev % k); line = " ".join(s) + "\n"
    f.write(line)
if f: f.write(tail); f.close()
print(f"{inp}: {iev+1} events -> {shard+1} shards")
