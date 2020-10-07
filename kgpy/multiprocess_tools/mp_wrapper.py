from itertools import repeat

#code stolen from https://stackoverflow.com/questions/45718523/pass-kwargs-to-starmap-while-using-pool-in-python

def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)