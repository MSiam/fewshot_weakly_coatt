import os
import sys
import tarfile
import argparse
import urllib.request as request
import pathlib
from zipfile import ZipFile

# The URL where the PASCAL VOC data can be downloaded.
# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
DATASET_URL_TRAIN = 'http://images.cocodataset.org/zips/train2014.zip'
DATASET_URL_VAL = 'http://images.cocodataset.org/zips/val2014.zip'
DATASET_URL_ANNOT = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'


def download_and_uncompress_dataset(dataset_dir: str, url: str):
    """Downloads PASCAL VOC and uncompresses it locally.
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """
    filename = url.split('/')[-1]
    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading {} {:.1f}%%'.format(
            filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    print()
    print("Downloading COCO from", url)
    print("Downloading COCO to", filepath)
    filepath, _ = request.urlretrieve(url, filepath, _progress)
    statinfo = os.stat(filepath)
    print()
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    #tarfile.open(filepath, 'r').extractall(dataset_dir)
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(filepath, 'r') as zipObj:
    # Extract all the contents of zip file in current directory
       zipObj.extractall(dataset_dir)
    print('Successfully downloaded and extracted COCO')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'pascal'),
        help='Path to the raw data')
    args = parser.parse_args()

#    download_and_uncompress_dataset(args.dataset_dir, DATASET_URL_TRAIN)
#    download_and_uncompress_dataset(args.dataset_dir, DATASET_URL_VAL)
    download_and_uncompress_dataset(args.dataset_dir, DATASET_URL_ANNOT)

