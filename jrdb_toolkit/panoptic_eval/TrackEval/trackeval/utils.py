
import os
import csv
import argparse
from collections import OrderedDict

import numpy as np


def init_config(config, default_config, name=None):
    """Initialise non-given config values with defaults"""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    if name and config['PRINT_CONFIG']:
        print('\n%s Config:' % name)
        for c in config.keys():
            print('%-20s : %-30s' % (c, config[c]))
    return config


def update_config(config):
    """
    Parse the arguments of a script and updates the config values for a given value if specified in the arguments.
    :param config: the config to update
    :return: the updated config
    """
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
    return config


def get_code_path():
    """Get base path where code is"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def validate_metrics_list(metrics_list):
    """Get names of metric class and ensures they are unique, further checks that the fields within each metric class
    do not have overlapping names.
    """
    metric_names = [metric.get_name() for metric in metrics_list]
    # check metric names are unique
    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException('Code being run with multiple metrics of the same name')
    fields = []
    for m in metrics_list:
        fields += m.fields
    # check metric fields are unique
    if len(fields) != len(set(fields)):
        raise TrackEvalException('Code being run with multiple metrics with fields of the same name')
    return metric_names


def write_summary_results(summaries, cls, output_folder):
    """Write summary results to file"""

    fields = sum([list(s.keys()) for s in summaries], [])
    values = sum([list(s.values()) for s in summaries], [])

    # In order to remain consistent upon new fields being adding, for each of the following fields if they are present
    # they will be output in the summary first in the order below. Any further fields will be output in the order each
    # metric family is called, and within each family either in the order they were added to the dict (python >= 3.6) or
    # randomly (python < 3.6).
    default_order = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA', 'HOTA(0)', 'LocA(0)',
                     'HOTALocA(0)', 'MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'CLR_TP', 'CLR_FN',
                     'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag', 'sMOTA', 'IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP',
                     'Dets', 'GT_Dets', 'IDs', 'GT_IDs']
    default_ordered_dict = OrderedDict(zip(default_order, [None for _ in default_order]))
    for f, v in zip(fields, values):
        default_ordered_dict[f] = v
    for df in default_order:
        if default_ordered_dict[df] is None:
            del default_ordered_dict[df]
    fields = list(default_ordered_dict.keys())
    values = list(default_ordered_dict.values())
    # replace / with _ in class name
    cls = cls.replace('/', '_')
    out_file = os.path.join(output_folder, cls + '_summary.txt')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(fields)
        writer.writerow(values)
    print('Summary results written to %s' % output_folder)

def safe_mean(data_list):
    clean_list = [x for x in data_list if not np.isnan(x)]
    if not clean_list:  # if clean_list is empty
        return np.nan
    return np.mean(clean_list)
def safe_sum(data_list):
    clean_list = [x for x in data_list if not np.isnan(x)]
    if not clean_list:  # if clean_list is empty
        return np.nan
    return np.sum(clean_list)
def write_per_sequence_detailed_results(details, output_folder):
    """Write per sequence results to file"""
    averaged_results = {}
    sum_fields = ["CLR_TP","MOTP_sum", "CLR_FN", "Frames", "CLR_Frames", "CLR_FP",'Frag', "IDSW", "MT", "PT", "ML", 'IDTP', 'IDFN', 'IDFP',  'Dets', 'GT_Dets', 'IDs', 'GT_IDs']
    all_fields = set()
    for sequence, classes, in details.items():
        if "COMBINED" in sequence:
            print("COMBINED")
        sequence_results = {}
        for classes_name, metrics in classes.items():
            for metric_name, metric_values in metrics.items():
                for key, value in metric_values.items():
                    all_fields.add(key)
                    if key not in sequence_results.keys():
                        sequence_results[key] = []
                    # Append value if it's a number (avoid non-numeric data)
                    if not np.isnan(value):
                        sequence_results[key].append(value)

        # Calculate the mean for each metric, avoiding NaNs
        for key, value_list in sequence_results.items():
            if key in sum_fields:
                sequence_results[key] = safe_sum(value_list)
            else:
                sequence_results[key] = safe_mean(value_list)
        averaged_results[sequence] = sequence_results

    fields = ['seq'] + list(sequence_results.keys())
    out_file = os.path.join(output_folder, 'per_sequence_detailed.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # check file exists
    if not os.path.isfile(out_file):
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for sequence in sorted(averaged_results.keys()):
                writer.writerow([sequence] + list(averaged_results[sequence].values()))
    print('Per sequence detailed results written to %s' % out_file)

def write_per_class_detailed_results(details, output_folder, split):
    """Write per class and category results to file, averaging over all sequences."""
    per_class_averaged_results = {}
    category_averaged_results = {k:{} for k in split.keys()}
    sum_fields = ["CLR_TP", "MOTP_sum", "CLR_FN", "Frames", "CLR_Frames", "CLR_FP", 'Frag', "IDSW", "MT", "PT", "ML",
                  'IDTP', 'IDFN', 'IDFP', 'Dets', 'GT_Dets', 'IDs', 'GT_IDs']
    all_fields = set()
    for sequence, classes, in details.items():
        for classes_name, metrics in classes.items():
            for metric_name, metric_values in metrics.items():
                for key, value in metric_values.items():
                    all_fields.add(key)
        break

    class_categories = split
    # Initialize temporary category data structure
    temp_category_data = {category: {field: [] for field in all_fields} for category in class_categories}

    # Accumulate all values per class and per category
    for sequence, classes in details.items():
        for class_name, metrics in classes.items():
            if 'cls_comb' in class_name:
                continue
            # Determine the category of the class
            category = next((cat for cat, classes in class_categories.items() if class_name in classes), None)

            # Initialize class in per_class_averaged_results if not present
            if class_name not in per_class_averaged_results:
                per_class_averaged_results[class_name] = {field: [] for field in all_fields}

            for metric_name, metric_values in metrics.items():
                for key, value in metric_values.items():
                    if key in all_fields:
                        # Append value to class and category
                        per_class_averaged_results[class_name][key].append(value)
                        if category:
                            temp_category_data[category][key].append(value)

    # Calculate the mean for each metric per class
    for class_name, metrics in per_class_averaged_results.items():
        for key, value_list in metrics.items():
            if key in sum_fields:
                per_class_averaged_results[class_name][key] = safe_sum(value_list)
            else:
                per_class_averaged_results[class_name][key] = safe_mean(value_list)

    # Calculate the mean for each metric per category
    for category, metrics in temp_category_data.items():
        for key, value_list in metrics.items():
            if key in sum_fields:
                category_averaged_results[category][key] = safe_sum(value_list)
            else:
                category_averaged_results[category][key] = safe_mean(value_list)

    # Combine class and category results
    combined_results = {**per_class_averaged_results, **category_averaged_results}

    # Determine the fields for the CSV based on the collected keys
    fields = ['class_name'] + list(next(iter(combined_results.values())).keys())
    out_file = os.path.join(output_folder, 'per_class_and_category_detailed.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for class_or_category, metrics in sorted(combined_results.items()):
            writer.writerow([class_or_category] + [metrics[key] for key in fields[1:]])
    print('Per class and category detailed results written to %s' % out_file)

def write_detailed_results(details, cls, output_folder):
    """Write detailed results to file"""
    sequences = details[0].keys()
    fields = ['seq'] + sum([list(s['COMBINED_SEQ'].keys()) for s in details], [])
    cls = cls.replace('/', '_')
    out_file = os.path.join(output_folder, cls + '_detailed.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for seq in sorted(sequences):
            if seq == 'COMBINED_SEQ':
                continue
            writer.writerow([seq] + sum([list(s[seq].values()) for s in details], []))
        writer.writerow(['COMBINED'] + sum([list(s['COMBINED_SEQ'].values()) for s in details], []))
    print('Detailed results written to %s' % output_folder)


def load_detail(file):
    """Loads detailed data for a tracker."""
    data = {}
    with open(file) as f:
        for i, row_text in enumerate(f):
            row = row_text.replace('\r', '').replace('\n', '').split(',')
            if i == 0:
                keys = row[1:]
                continue
            current_values = row[1:]
            seq = row[0]
            if seq == 'COMBINED':
                seq = 'COMBINED_SEQ'
            if (len(current_values) == len(keys)) and seq != '':
                data[seq] = {}
                for key, value in zip(keys, current_values):
                    data[seq][key] = float(value)
    return data


class TrackEvalException(Exception):
    """Custom exception for catching expected errors."""
    ...

