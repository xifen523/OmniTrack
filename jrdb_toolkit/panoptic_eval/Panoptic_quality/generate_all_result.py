import os
import argparse
import json
import csv
from decimal import Decimal, getcontext
TEST = [
    'cubberly-auditorium-2019-04-22_1',
    'discovery-walk-2019-02-28_0',
    'discovery-walk-2019-02-28_1',
    'food-trucks-2019-02-12_0',
    'gates-ai-lab-2019-04-17_0',
    'gates-basement-elevators-2019-01-17_0',
    'gates-foyer-2019-01-17_0',
    'gates-to-clark-2019-02-28_0',
    'hewlett-class-2019-01-23_0',
    'hewlett-class-2019-01-23_1',
    'huang-2-2019-01-25_1',
    'huang-intersection-2019-01-22_0',
    'indoor-coupa-cafe-2019-02-06_0',
    'lomita-serra-intersection-2019-01-30_0',
    'meyer-green-2019-03-16_1',
    'nvidia-aud-2019-01-25_0',
    'nvidia-aud-2019-04-18_1',
    'nvidia-aud-2019-04-18_2',
    'outdoor-coupa-cafe-2019-02-06_0',
    'quarry-road-2019-02-28_0',
    'serra-street-2019-01-30_0',
    'stlc-111-2019-04-19_1',
    'stlc-111-2019-04-19_2',
    'tressider-2019-03-16_2',
    'tressider-2019-04-26_0',
    'tressider-2019-04-26_1',
    'tressider-2019-04-26_3'
]

def report_average_pq(pred_dir,OW):
    single_result="All"
    file_numbers = len(TEST)
    all_res={}
    for file_name in TEST:
            with open(os.path.join(pred_dir,file_name+"_pq.json"),"r") as readfile:
                all_res[file_name] = json.load(readfile)
    from decimal import Decimal

    if OW:
        output_result = {
            "All": {
                "pq": Decimal("0.0"),
                "sq": Decimal("0.0"),
                "rq": Decimal("0.0"),
            },
            "Things": {
                "pq": Decimal("0.0"),
                "sq": Decimal("0.0"),
                "rq": Decimal("0.0"),
            },
            "Stuff": {
                "pq": Decimal("0.0"),
                "sq": Decimal("0.0"),
                "rq": Decimal("0.0"),
            },
            "Known": {
                "pq": Decimal("0.0"),
                "sq": Decimal("0.0"),
                "rq": Decimal("0.0"),
            },
            "Unknown": {
                "pq": Decimal("0.0"),
                "sq": Decimal("0.0"),
                "rq": Decimal("0.0"),
            }
        }
    else:
        output_result = {
            "All": {
                "pq": Decimal("0.0"),
                "sq": Decimal("0.0"),
                "rq": Decimal("0.0"),
            },
            "Things": {
                "pq": Decimal("0.0"),
                "sq": Decimal("0.0"),
                "rq": Decimal("0.0"),
            },
            "Stuff": {
                "pq": Decimal("0.0"),
                "sq": Decimal("0.0"),
                "rq": Decimal("0.0"),
            }
        }

    
    for metric_name,metric_items in output_result.items():
        count=0
        for metric_item in metric_items:
            for file_name,single_item in all_res.items():
                #single_item[metric_name][metric_item] = single_item[metric_name][metric_item]*100
                output_result[metric_name][metric_item]+=Decimal(single_item[metric_name][metric_item])

            output_result[metric_name][metric_item] = (output_result[metric_name][metric_item]/ file_numbers)
    return output_result,all_res
                # pq,sq,rq


def saving_csv(average_results,persequence_result,OW,pred_path):
    pred_path = os.path.abspath(os.path.join(pred_path, os.pardir))

    #print("file_path = os.path.join(pred_path, 'pq', 'average_metrics.csv')",os.path.join(pred_path, 'pq', 'average_metrics.csv'))
    flattened_data = []
    if OW:
        header = ['Category', 'PQ', 'SQ', 'RQ', 'PQ_Things', 'SQ_Things', 'RQ_Things', 'PQ_Stuff', 'SQ_Stuff', 'RQ_Stuff', 'PQ_Known', 'SQ_Known', 'RQ_Known', 'PQ_Unknown', 'SQ_Unknown', 'RQ_Unknown']
        mapped_name = ["All", "Things", "Stuff", "Known", "Unknown"]
        flattened_data = []
        for file_name in persequence_result.keys():
            temp_data = [file_name]
            for name in mapped_name:
                temp_data.extend([
                    f"{persequence_result[file_name][name]['pq']:.3f}",
                    f"{persequence_result[file_name][name]['sq']:.3f}",
                    f"{persequence_result[file_name][name]['rq']:.3f}"
                ])
            flattened_data.append(temp_data)
        temp_data = ["averaged"]
        for name in mapped_name:
            temp_data.extend([
                f"{average_results[name]['pq']:.3f}",
                f"{average_results[name]['sq']:.3f}",
                f"{average_results[name]['rq']:.3f}"
            ])
        flattened_data.append(temp_data)
    else:
        header = ['Category', 'PQ', 'SQ', 'RQ', 'PQ_Things', 'SQ_Things', 'RQ_Things', 'PQ_Stuff', 'SQ_Stuff', 'RQ_Stuff']
        mapped_name = ["All", "Things", "Stuff"]
        flattened_data = []
        for file_name in persequence_result.keys():
            temp_data = [file_name]
            for name in mapped_name:
                temp_data.extend([
                    f"{persequence_result[file_name][name]['pq']:.3f}",
                    f"{persequence_result[file_name][name]['sq']:.3f}",
                    f"{persequence_result[file_name][name]['rq']:.3f}"
                ])
            flattened_data.append(temp_data)
        temp_data = ["averaged"]
        for name in mapped_name:
            temp_data.extend([
                f"{average_results[name]['pq']:.3f}",
                f"{average_results[name]['sq']:.3f}",
                f"{average_results[name]['rq']:.3f}"
            ])
        flattened_data.append(temp_data)

                # CSV file path
    os.makedirs(os.path.join(pred_path, 'pq'), exist_ok=True)
    file_path = os.path.join(pred_path, 'pq', 'average_metrics.csv')

    # Writing to CSV
    with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Writing the header
            writer.writerow(header)
            # Writing the data
            writer.writerows(flattened_data)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--pred_dir', type=str, default=None,
                       help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--OW', action='store_true',
                        help="Open World Metric")
    parser.add_argument('--CSV', action='store_true',
                        help="Open World Metric")
    args = parser.parse_args()
    average_results, persequence_result= report_average_pq(args.pred_dir,args.OW)
    if args.CSV:
        saving_csv(average_results, persequence_result,args.OW,args.pred_dir)
    #results_record_name = os.path.join(args.pred_dir,"pq.json")
    #with open(results_record_name, 'w') as json_file:
    #     json.dump(results, json_file, indent=4)
    