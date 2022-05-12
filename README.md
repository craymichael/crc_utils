# crc_utils

## Install Scripts

This will install `crc_blame.py`.

```shell
git clone <this repository>
cd crc_utils/
./install
```

After installation, make sure that `~/.local/bin/` is in your `$PATH`.
You can ensure that it is by adding the following line to your `~/.bashrc`:

```shell
export PATH="~/.local/bin/:$PATH"
```

## Usage

For `crc_blame.py` usage:

```shell
crc_blame.py -h
```

For example:

```shell
$ crc_blame.py -u $USER
Top 10 slots by user
 tot_gpu  tot_jobs  tot_slots      user
       ...

Top 10 GPUs by user
 tot_gpu  tot_jobs  tot_slots      user
      ...

Top 10 free slots by node (x/x free total)
                    queuename  avail_slots  used_slots  avail_gpus  used_gpus
 ...

Top 10 free GPUs by node (x/x free total)
                    queuename  avail_slots  used_slots  avail_gpus  used_gpus
 ...
```

## Troubleshooting

If you get the following error (or another Python/dependency-related error):
`You must have pandas installed to use this script!`,
make sure that you first load the `python` module before running scripts, i.e.:

```shell
module load python
```

You can add this to your `~/.bashrc` too so you don't have to run it every time.

Alternatively, you may have run the script while in a virtual environment without
the proper dependencies installed. To fix, you can either 1) run the script outside
of the virtual environment, or 2) you can install the dependency `pandas` into
your virtual environment.
