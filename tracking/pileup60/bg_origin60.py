"""Origin of the tracks tagged in the no-quirk mu60 sample: majority particle traced in the ORIGINAL truth (quirks kept)."""
import json, numpy as np, polars as pl
Q="/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data"
W="/pscratch/sd/d/danieltm/quirk_pu60/lambda/m200_L774_mu60_noquirk/bg/qw40_cc0.5"
tj=json.load(open(f"{Q}/models/tagger/b3_pair60_50k_qw40_cc0.5/tagger.json"))
qj=json.load(open(f"{Q}/models/quality_pu60/pair60_50k_qw40_cc0.5/roc.json"))
tc=tj["pair_good"]["cut_from_valB"]; q02=qj["operating_points"]["fake_0.02"]["cut_valB"]
tr=pl.read_parquet(W+"/tracks_test.parquet").join(pl.read_parquet(W+"/bg_scores.parquet"),on=["event_id","track_id"])
tag=tr.filter(pl.col("quirk_tag")>=tc).with_columns(q02=pl.col("quality")>=q02)
P=pl.concat([pl.read_parquet(f"{Q}/parquet_r1_mu60/m200_L774_mu60/truth/particles/quirks.m200_L774_mu60.v1.truth.particles.events{b}-{b+999}.parquet",
   columns=["event_id","particle_id","pdg_id","parent_id","vertex_primary","primary","px","py","pz","vx","vy","vz"]) for b in (60000,61000)]).explode(pl.exclude("event_id"))
print("quirk vertex_primary values:",P.filter(pl.col("pdg_id").abs()==10000100)["vertex_primary"].unique().to_list())
rows=[]
for e,g in P.group_by("event_id"):
    pid=g["particle_id"].to_numpy(); par=g["parent_id"].to_numpy(); isq=np.abs(g["pdg_id"].to_numpy())==10000100
    idx={int(p):i for i,p in enumerate(pid)}; qd=np.zeros(len(pid),bool)
    for i in range(len(pid)):
        j=i; seen=0
        while seen<200:
            j=idx.get(int(par[j]),-1) if par[j]>=0 else -1
            if j<0: break
            if isq[j]: qd[i]=True; break
            seen+=1
    rows.append(g.with_columns(qdesc=pl.Series(qd)))
P=pl.concat(rows).with_columns(pt=(pl.col("px")**2+pl.col("py")**2).sqrt(),rprod=(pl.col("vx")**2+pl.col("vy")**2).sqrt())
qvp=set(P.filter(pl.col("pdg_id").abs()==10000100)["vertex_primary"].unique().to_list())
T=tag.join(P.rename({"particle_id":"maj_pid"}),on=["event_id","maj_pid"],how="left").with_columns(
  origin=pl.when(pl.col("maj_pid").is_null()|pl.col("pdg_id").is_null()).then(pl.lit("no_truth"))
   .when(pl.col("qdesc")).then(pl.lit("quirk_descendant"))
   .when(pl.col("vertex_primary").is_in(list(qvp))).then(pl.lit("quirk_vertex_other"))
   .otherwise(pl.lit("pileup")))
for lab,d in (("tagger",T),("tagger+fake_0.02",T.filter("q02"))):
    print(f"\n== {lab}: {d.height} tracks")
    print(d.group_by("origin").agg(n=pl.len(),pt_med=pl.col("pt").median(),pt_q10=pl.col("pt").quantile(.1),pt_q90=pl.col("pt").quantile(.9),
          rprod_med=pl.col("rprod").median(),purity_med=pl.col("purity").median(),nhits_med=pl.col("n_hits").median()).sort("n",descending=True))
    print("pT bins (GeV):",d.select(pl.col("pt").cut([0.5,1,2,5,10],labels=["<0.5","0.5-1","1-2","2-5","5-10",">10"]).value_counts(sort=True)).unnest("pt").to_dicts())
    print("top |pdg|:",d.group_by(pl.col("pdg_id").abs()).len().sort("len",descending=True).head(6).to_dicts())
T.write_parquet(W+"/bg_tagged_origin.parquet")
