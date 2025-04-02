import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'trajtrack')))
import trackeval  # noqa: E402

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.JRDB3DTraj.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['DE','OSPA','Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.JRDB3DTraj(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.DE,trackeval.metrics.OSPA,trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            if metric.get_name()=='CLEAR':
                metrics_list.append(metric(config={'THRESHOLD': 0.3,'PRINT_CONFIG': True}))
            else:
                metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    print('len',len(metrics_list))
    evaluator.evaluate(dataset_list, metrics_list,is_3d=True)
    
def evaluate(tracker_path):
    freeze_support()
    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.JRDB3DTraj.get_default_dataset_config()
    default_metrics_config = {'METRICS':  ['DE','OSPA','Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    config['TRACKERS_FOLDER']=tracker_path
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.JRDB3DTraj(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.DE,trackeval.metrics.OSPA,trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            if metric.get_name()=='CLEAR':
                metrics_list.append(metric(config={'THRESHOLD': 0.3,'PRINT_CONFIG': True}))
            else:
                metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    print('here at least 3d')
    return evaluator.evaluate(dataset_list, metrics_list,is_3d=True)
