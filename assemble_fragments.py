import argparse
import importlib
import pdb
from runner import fragment_reassembler

def main(args):

    print_line_length = 65
    module_name = f"configs.{args.cfg}"
    cfg = importlib.import_module(module_name)

    print("\nWill try to assemble:")
    for i, broken_objects in enumerate(cfg.data_list):
        category = broken_objects["category"]
        fracture = broken_objects["fracture"]
        name = f"{category}_{fracture}"
        print(f"{i}) {name}")

    print()
    for i, broken_objects in enumerate(cfg.data_list):

        print('#' * print_line_length)
        category = broken_objects["category"]
        fracture = broken_objects["fracture"]
        name = f"{category}_{fracture}"
        print(f'Current broken object: {fracture} ({category})')
        fr_ass = fragment_reassembler(broken_objects, cfg.variables_as_list, cfg.pipeline_parameters, name, show_results = True, save_results=True)
        print('-' * print_line_length)
        # load object and move them (challenge_R_T) to have something to solve (without R and T they are loaded in the correct position)
        fr_ass.load_objects()

        # set output directory, save fragments and information there
        fr_ass.set_output_dir(cfg.output_dir)
        fr_ass.save_fragments(cfg.name)
        fr_ass.save_info()

        # 3.1) breaking curves
        print('-' * print_line_length)
        fr_ass.detect_breaking_curves()
        fr_ass.save_breaking_curves()

        # 3.2) segmentation
        print('-' * print_line_length)
        fr_ass.segment_regions()
        fr_ass.save_segmented_regions()

        # 3.3) registration
        print('-' * print_line_length)
        fr_ass.register_segmented_regions()
        fr_ass.save_registration_results()
        fr_ass.save_registered_pcls()

        # TODO: untested! probably not working smooth without debugging
        # if we have ground truth, evaluate it
        # if 'solution' in broken_objects.keys():
        #     print('-' * print_line_length)
        #     fr_ass.evaluate_against_gt()
        #     fr_ass.save_evaluation_results()

        print('#' * print_line_length)

    pdb.set_trace()
    return 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reassembling broken objects')
    parser.add_argument('--cfg', type=str, default='assemble_cfg', help='config file (.py)')
    args = parser.parse_args()
    main(args)
