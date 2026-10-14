"""Rewrite a HepMC3 ASCII file as HepMC2 ASCII (run with an LCG view whose HepMC3 reads the input).

Events are renumbered from 0: quirk-gen numbers them from 1, and ACTS' HepMC3Reader (pileup merge)
stops with "Event number mismatch" unless the first event is 0.
"""
import sys
from pyHepMC3 import HepMC3 as hm
src, dst = sys.argv[1], sys.argv[2]
r = hm.ReaderAscii(src); w = hm.WriterAsciiHepMC2(dst); n = 0
while True:
    e = hm.GenEvent()
    if not r.read_event(e) or r.failed():
        break
    e.set_event_number(n)
    w.write_event(e); n += 1
w.close(); r.close()
print(f"converted {n} events: {src} -> {dst}")
