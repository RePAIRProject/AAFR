import argparse
import importlib
import pdb 
from runner import fragment_reassembler

def main(args):

    print_line_length = 65
    module_name = f"configs.{args.cfg}"
    cfg = importlib.import_module(module_name)
    
    for i, broken_objects in enumerate(cfg.data_list):

        print('\n')
        print('-' * print_line_length)
        category = broken_objects["path_obj1"].split("/")[-3]
        fracture = broken_objects["path_obj1"].split("/")[-2]
        print(f'Current broken object: {fracture} ({category})')
        fr_ass = fragment_reassembler(broken_objects, cfg.pipeline_parameters, cfg.name, show_results = True, save_results=True)
        print('-' * print_line_length)
        fr_ass.load_objects()
        fr_ass.set_output_dir(cfg.output_dir)
        fr_ass.save_fragments(cfg.name)
        fr_ass.save_info()
        print('-' * print_line_length)
        fr_ass.detect_breaking_curves()
        fr_ass.save_breaking_curves()
        print('-' * print_line_length)
        fr_ass.segment_regions()
        fr_ass.save_segmented_regions()
        print('-' * print_line_length)

    pdb.set_trace()
    return 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reassembling broken objects')
    parser.add_argument('--cfg', type=str, default='base', help='config file (.py)')
    args = parser.parse_args()
    main(args)