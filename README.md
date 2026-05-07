# DiversePlace Implememtation for NIPS2026 Submission

## Setup

```bash
conda env create -f environment.yml
conda activate macro_place
```

Download the public ISPD 2005 placement benchmarks from the official contest page:

https://www.ispd.cc/contests/05/contest.htm

Place the extracted benchmark directories next to this repo so the configs can resolve them:

```bash
mkdir -p ../ispd2005free
cd ../ispd2005free

wget -nc http://www.ispd.cc/contests/05/ispd05-contest/adaptec1.tar.gz
wget -nc http://www.ispd.cc/contests/05/ispd05-contest/adaptec3.tar.gz

for benchmark in adaptec2 adaptec4 bigblue1 bigblue2 bigblue3 bigblue4; do
  wget -nc "http://www.ispd.cc/contests/05/ispd05-contest/benchmarks/${benchmark}.tar.gz"
done

for archive in *.tar.gz; do
  tar -xzf "${archive}"
done
cd -
```

The expected layout is:

```text
../ispd2005free/adaptec1
../ispd2005free/adaptec2
../ispd2005free/adaptec3
../ispd2005free/adaptec4
../ispd2005free/bigblue1
../ispd2005free/bigblue2
../ispd2005free/bigblue3
../ispd2005free/bigblue4
```

## Train

Run from the repo root:

```bash
python main.py benchmark@_global_=adaptec1
python main.py benchmark@_global_=adaptec2
python main.py benchmark@_global_=adaptec3
python main.py benchmark@_global_=adaptec4
python main.py benchmark@_global_=bigblue1
python main.py benchmark@_global_=bigblue2
python main.py benchmark@_global_=bigblue3
python main.py benchmark@_global_=bigblue4
```
