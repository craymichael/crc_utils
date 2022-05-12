#!/usr/bin/env python3
import subprocess
import re
import os
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description='Show top users of CRC resources.'
)

def queue_t(value):
    if not value.startswith('@'):
        raise argparse.ArgumentTypeError('Queue (host group) name must start '
                                         'with @')
    return value

parser.add_argument(
    '--queue', '-q', help=('Show only resources for this queue (host group), '
                           'otherwise show for all queues. Tip: you can '
                           'print all queues using the command `qconf '
                           '-shgrpl`'),
    type=queue_t
)
parser.add_argument(
    '--user', '-u', help=('Show only resources available to this user, '
                          'otherwise show for all users.'),
)
parser.add_argument(
    '--top-k', '-k', help='Show k top users of resources (Default: 10)',
    type=int, default=10
)

args = parser.parse_args()

try:
    import pandas as pd
except ImportError:
    import sys

    sys.exit('You must have pandas installed to use this script! Running '
             'Python {}\nExecutable: {}\nLibrary paths: {}'.format(
                 sys.version, sys.executable, sys.path))

qstat_cmd = ['qstat', '-F', '-r']
if args.user:
    qstat_cmd.extend(['-U', args.user])

result = subprocess.run(qstat_cmd, capture_output=True)
data = result.stdout.decode('UTF-8')
# remove pending jobs
data = data.split('#' * 79 + '\n' + ' - PENDING JOBS')[0]
# split by section
data = data.split('-' * 81)

# header validation
expected_header = ['queuename', 'qtype', 'resv/used/tot.', 'load_avg', 'arch',
                   'states']
header = data[0].split()
assert len(header) == 6, data[0]
assert header == expected_header

job_header = ['id', 'priority', 'name', 'user', 'status', 'date', 'time',
              'slots']

def handle_hard_resource(value, job, cur_value=None):
    value = value.strip()
    if value:
        v_name_val = value.split('=', 1)
        assert len(v_name_val) == 2, value
        v_name, v_value = v_name_val
        v_value = v_value.rsplit(' (', 1)[0]
        if cur_value:
            cur_value[v_name] = v_value
        else:
            cur_value = {v_name: v_value}
        job['hard_resources'] = cur_value
        if v_name == 'gpu_card':  # special case
            job[v_name] = v_value
    else:
        job['hard_resources'] = job.get('hard_resources', {})

nodes = []
jobs = []
for section in data[1:]:
    sec_data = {}
    sec_jobs = []
    for line in section.split('\n'):
        # print('{: 3d} {}'.format(len(line), line))  # debugging
        if not line:
            continue
        if re.match(r'^\t[^\s]', line):  # node resources
            name_val = line.split('=', 1)
            assert len(name_val) == 2, line
            name, value = name_val
            sec_data[name.strip()] = value.strip()
        elif re.match(r'^ {7}[^\s]', line):  # job details (under a job)
            assert sec_jobs
            job = sec_jobs[-1]
            name_val = line.split(':', 1)
            assert len(name_val) == 2, line
            name, value = name_val[0].strip(), name_val[1].strip()
            name = name.lower().replace(' ', '_')
            if name == 'hard_resources':
                handle_hard_resource(value, job)
            else:
                job[name] = value
        elif re.match(r'^ {2,9}\d', line):  # job
            job = OrderedDict()
            for name, value in zip(job_header, line.split()):
                job[name] = value.strip()
            sec_jobs.append(job)
        elif re.match(r'^ {25}[^\s]', line):  # continuation of job details
            # assume grouped with last header added to job description
            # also with format x=y
            assert sec_jobs
            job = sec_jobs[-1]
            last_headers = [*job.keys()][-2:]
            if (last_headers[-1] == 'hard_resources' or
                    (last_headers[-1] == 'gpu_card' and
                     last_headers[-2] == 'hard_resources')):
                handle_hard_resource(value, job, job['hard_resources'])
            else:
                raise NotImplementedError(
                    'Job header {}'.format(last_header))
        elif line[0] != ' ':  # node information
            for i, (name, value) in enumerate(zip(header, line.split())):
                if i == 2:  # resv/used/tot.
                    values = value.strip().split('/')
                    assert len(values) == 3, (name, value)
                    sec_data.update(resv=values[0], used=values[1],
                                    tot=values[2])
                else:
                    sec_data[name] = value.strip()
        else:  # unknown
            raise RuntimeError('Unknown line format:\n\t{}'.format(line))
    nodes.append(sec_data)
    jobs.extend(sec_jobs)

def cast(df, name, type_):
    df.loc[:, name] = df.loc[:, name].astype(type_)

df_nodes = pd.DataFrame(nodes)
df_jobs = pd.DataFrame(jobs)

if args.queue is not None:
    result = subprocess.run(['qconf', '-shgrp_tree', args.queue],
                            capture_output=True)
    relevant_nodes = [
        q for q in
        map(str.strip, result.stdout.decode('UTF-8').split())
        if not q.startswith('@')
    ]
    df_jobs = df_jobs.loc[df_jobs['master_queue'].apply(
        lambda x: x.split('@')[1] in relevant_nodes)]
    df_nodes = df_nodes.loc[df_nodes['queuename'].apply(
        lambda x: x.split('@')[1] in relevant_nodes)]

cast(df_nodes, 'used', float)
cast(df_nodes, 'resv', float)
cast(df_nodes, 'tot', float)
cast(df_nodes, 'hc:gpu_card', float)
cast(df_nodes, 'hl:m_gpu', float)

cast(df_jobs, 'slots', float)
cast(df_jobs, 'gpu_card', float)

try:
    disp_width = os.get_terminal_size().columns
except OSError:
    disp_width = None

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', disp_width):
    '''
    for queue, df_q in df_jobs.groupby('master_queue'):
        hostname = queue.split('@', 1)[1]
        df_n = df_nodes[df_nodes['qf:hostname'] == hostname]
        assert len(df_n) == 1, (queue, hostname)
        s_n = df_n.iloc[0]

        unavail = s_n['used'] + s_n['resv']
        in_use = df_q['slots'].sum()

        assert unavail == in_use, (unavail, '|', in_use, '/', s_n['tot'])
    '''

    job_stats = []
    for user, df_u in df_jobs.groupby('user'):
        row = dict(
            user=user,
            tot_slots=df_u['slots'].sum(),
            tot_gpu=df_u['gpu_card'].sum(),
            tot_jobs=len(df_u),
        )
        job_stats.append(row)

    job_stats = pd.DataFrame(job_stats)
    cast(job_stats, 'tot_gpu', int)
    cast(job_stats, 'tot_slots', int)

    node_stats = pd.DataFrame()
    node_stats['queuename'] = df_nodes['queuename']
    node_stats['avail_slots'] = (df_nodes['tot'] - df_nodes['used']
                                 - df_nodes['resv'])
    node_stats['used_slots'] = df_nodes['used'] + df_nodes['resv']
    node_stats['avail_gpus'] = df_nodes['hc:gpu_card']
    node_stats['used_gpus'] = df_nodes['hl:m_gpu'] - df_nodes['hc:gpu_card']
    node_stats.fillna({
        'avail_gpus': 0,
        'used_gpus': 0,
    }, inplace=True)
    cast(node_stats, 'avail_slots', int)
    cast(node_stats, 'used_slots', int)
    cast(node_stats, 'avail_gpus', int)
    cast(node_stats, 'used_gpus', int)

    print('Top {} slots by user'.format(args.top_k))
    print(job_stats.sort_values(by=['tot_slots', 'tot_gpu'],
                                ascending=False).head(args.top_k).reset_index(
                                drop=True).to_string(index=False))

    print('\nTop {} GPUs by user'.format(args.top_k))
    print(job_stats.sort_values(by=['tot_gpu', 'tot_slots'],
                                ascending=False).head(args.top_k).reset_index(
                                drop=True).to_string(index=False))

    print('\nTop {} free slots by node ({}/{} free total)'.format(
        args.top_k, node_stats['avail_slots'].sum(),
        node_stats['used_slots'].sum() + node_stats['avail_slots'].sum()))
    print(node_stats.sort_values(by=['avail_slots', 'avail_gpus'],
                                 ascending=False).head(args.top_k).reset_index(
                                 drop=True).to_string(index=False))

    print('\nTop {} free GPUs by node ({}/{} free total)'.format(
        args.top_k, node_stats['avail_gpus'].sum(),
        node_stats['used_gpus'].sum() + node_stats['avail_gpus'].sum()))
    print(node_stats.sort_values(by=['avail_gpus', 'avail_slots'],
                                 ascending=False).head(args.top_k).reset_index(
                                 drop=True).to_string(index=False))
