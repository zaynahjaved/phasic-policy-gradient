from mpi4py import MPI
import argparse
from . import torch_util as tu
import torch as th
from . import logger
from .envs import get_venv
from .log_save_helper import LogSaveHelper
from .roller import Roller

def test_fn(env_name="fruitbot",
    distribution_mode="easy",
    interacts_total=1000000,
    num_envs=64,
    model_path='',
    log_dir='/tmp/test'
    ):

    comm = MPI.COMM_WORLD
    tu.setup_dist(comm=comm)

    if log_dir is not None:
        format_strs = ['csv', 'stdout'] if comm.Get_rank() == 0 else []
        logger.configure(comm=comm, dir=log_dir, format_strs=format_strs)

    venv = get_venv(num_envs=num_envs, env_name=env_name, distribution_mode=distribution_mode)

    model = th.load(model_path, map_location=th.device('cpu'))

    model.to(tu.dev())
    logger.log(tu.format_model(model))
    tu.sync_params(model.parameters())


    log_save_opts={"save_mode": "last"}
    learn_state = None
    nstep = 256

    while True:

        learn_state = learn_state or {}
        ic_per_step = venv.num * comm.size * nstep

        roller = learn_state.get("roller") or Roller(
            act_fn=model.act,
            venv=venv,
            initial_state=model.initial_state(venv.num),
            keep_buf=100,
            keep_non_rolling=log_save_opts.get("log_new_eps", False),
        )

        lsh = learn_state.get("lsh") or LogSaveHelper(
            ic_per_step=ic_per_step, model=model, comm=comm, **log_save_opts
        )

        curr_interact_count = learn_state.get("curr_interact_count") or 0

        while curr_interact_count < interacts_total:
            seg = roller.multi_step(nstep)
            lsh.gather_roller_stats(roller)
            lsh()

            curr_interact_count += ic_per_step

        learn_state = dict(
            roller=roller,
            lsh=lsh,
            curr_interact_count=curr_interact_count,
        )

        if learn_state["curr_interact_count"] >= interacts_total:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    args=parser.parse_args()
    test_fn(env_name='fruitbot',model_path=args.model_path)
