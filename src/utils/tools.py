import os
import glob
import json
import zipfile


def zip_file(file_path, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)

    zipw = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(file_path):
        for f in files:
            zipw.write(os.path.join(root, f))

    zipw.close()


def append_json(path, data, mode='a'):
    with open(path, mode) as f:
        for image_id, vector in data.items():
            tmp = {'key': image_id, 'value': vector}
            f.write(json.dumps(tmp) + '\n')


def get_lastest_index(data_dir):
    items = os.listdir(data_dir)
    max_index = 0
    for item in items:
        index = int(os.path.splitext(item)[0].split('_')[1])
        if index > max_index:
            max_index = index
    return max_index


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors,
                                     all_tensor_names=False,
                                     count_exclude_pattern=""):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
        file_name: Name of the checkpoint file.
        tensor_name: Name of the tensor in the checkpoint file to print.
        all_tensors: Boolean indicating whether to print all tensors.
        all_tensor_names: Boolean indicating whether to print all tensor names.
        count_exclude_pattern: Regex string, pattern to exclude tensors.
    """
    from tensorflow.python import pywrap_tensorflow

    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors or all_tensor_names:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print("tensor_name: ", key)
                if all_tensors:
                    print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))

            # Count total number of parameters
            print("# Total number of params: %d" % _count_total_params(
                reader, count_exclude_pattern=count_exclude_pattern))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        if ("Data loss" in str(e) and
                any(e in file_name for e in [".index", ".meta", ".data"])):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
    It's likely that this is a V2 checkpoint and you need to provide the
    filename *prefix*.  Try removing the '.' and extension.  Try:
    inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))


if __name__ == '__main__':
    print_tensors_in_checkpoint_file(
        'train_log/vqvae-org-128x128/model-200000', None, None)
