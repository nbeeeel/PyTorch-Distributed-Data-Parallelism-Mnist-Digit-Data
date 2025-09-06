import os, json, socket, time

def setup_env():
    os.environ['MPLCONFIGDIR'] = os.environ.get(
        'MPLCONFIGDIR', f'/tmp/matplotlib-{os.environ.get("SLURM_JOBID", "default")}'
    )

    master_addr = os.environ.get('MASTER_ADDR', '192.168.20.15')
    master_port = os.environ.get('MASTER_PORT', '29500')
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    tf_config = {
        'cluster': {'worker': [f'{master_addr}:{master_port}' for _ in range(world_size)]},
        'task': {'type': 'worker', 'index': rank}
    }

    print(f"RANK {rank} on {socket.gethostname()}: Setting TF_CONFIG: {json.dumps(tf_config)}")
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    os.environ['TF_GRPC_DEFAULT_OPTIONS'] = 'grpc.keepalive_time_ms=30000,grpc.keepalive_timeout_ms=15000'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if os.environ.get('NCCL_DEBUG') == 'INFO' else '2'

    if rank == 0:
        print(f"RANK {rank} leader waiting 5s for workers...")
        time.sleep(5)

    return rank, master_addr, master_port
