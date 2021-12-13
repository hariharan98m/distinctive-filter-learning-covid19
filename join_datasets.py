import pandas as pd
from pathlib import Path
import shutil
import re
import pydicom as dicom
import cv2
import pdb
import png

def ensure_exists(path):
    Path.mkdir(path, exist_ok=True, parents=True)

class AGChung:
    def __init__(self, path):
        self.path = path
        self.metadata = pd.read_csv(str(Path(path) / 'metadata.csv'), encoding = 'ISO-8859-1')

    def create_dataset(self, savepath):
        covid19_dir = Path(savepath) / 'covid19'
        ensure_exists(covid19_dir)  # ensure path exists.

        covid19_xray_samples = self.metadata[(self.metadata['finding'] == 'COVID-19')][['patientid']]
        ext_fn = lambda filename: filename + ('.png' if (Path(self.path) / 'images' / (filename + '.png')).exists() else '.jpg')
        covid19_xray_samples['filename'] = covid19_xray_samples['patientid'].apply(ext_fn)
        covid19_xray_samples['patientid'] = covid19_xray_samples['patientid'].apply(lambda id: re.search('COVID-\d{5}', id).group(0))
        print('AGChung unique patients: ', len(covid19_xray_samples['patientid'].unique()), ', total samples: ', len(covid19_xray_samples))

        for _, sample in covid19_xray_samples.iterrows():
            pid, filename = sample['patientid'], sample['filename']
            new_name = 'agchung_%s_%s' % (pid, filename)
            shutil.copy(str(Path(self.path) / 'images' / filename), str(covid19_dir / new_name))

class JCohen:
    def __init__(self, path):
        self.path = path
        self.metadata = pd.read_csv(str(Path(path) / 'metadata.csv'))
        self.covid19_urls = set()

    def create_dataset(self, savepath, only_urls = False):
        covid19_dir = Path(savepath) / 'covid19'
        ensure_exists(covid19_dir)  # ensure path exists.

        covid19_xray_samples = self.metadata[(self.metadata['finding'] == 'COVID-19') &
                                              (self.metadata['view'].isin(["PA", "AP", "AP Supine", "AP semi erect", "AP erect"])) &
                                              (self.metadata['modality'] == 'X-ray')][['patientid', 'filename', 'url']]
        self.covid19_urls = set(covid19_xray_samples['url'].tolist())
        if only_urls:
            return self.covid19_urls
        print('Cohen unique patients: ', len(covid19_xray_samples['patientid'].unique()), ', total samples: ', len(covid19_xray_samples))
        for _, sample in covid19_xray_samples.iterrows():
            pid, filename = sample['patientid'], sample['filename']
            new_name = 'cohen_%s_%s' % (pid, filename)
            shutil.copy(str(Path(self.path) / 'images' / filename), str(covid19_dir / new_name))

class ActualMed:
    def __init__(self, path):
        self.path = path
        self.metadata = pd.read_csv(str(Path(path) / 'metadata.csv'))

    def create_dataset(self, savepath):
        covid19_dir = Path(savepath) / 'covid19'
        ensure_exists(covid19_dir)  # ensure path exists.

        covid19_xray_samples = self.metadata[(self.metadata['finding'] == 'COVID-19')][['patientid', 'imagename']]
        print('ActualMed unique patients: ', len(covid19_xray_samples['patientid'].unique()), ', total samples: ', len(covid19_xray_samples))

        for _, sample in covid19_xray_samples.iterrows():
            pid, filename = sample['patientid'], sample['imagename']
            new_name = 'actualmed_%s_%s' % (pid, filename)
            shutil.copy(str(Path(self.path) / 'images' / filename), str(covid19_dir / new_name))


class SIRM:
    def __init__(self, path, cohen_urls):
        self.path = path
        self.discard_pids = ['100', '101', '102', '103', '104', '105',
                   '110', '111', '112', '113', '122', '123',
                   '124', '125', '126', '217']
        self.cohen_urls = cohen_urls
        self.covid19_metadata = pd.read_excel(str(Path(path) / 'COVID-19.metadata.xlsx'))

    def create_dataset(self, savepath):
        pdb.set_trace()
        covid19_dir = Path(savepath) / 'covid19'
        ensure_exists(covid19_dir)  # ensure path exists.

        covid19_xray_samples = self.covid19_metadata[~(self.covid19_metadata['URL'].isin(self.cohen_urls))]
        covid19_xray_samples['patientid'] = covid19_xray_samples['FILE NAME'].apply(lambda filename: re.search('\((\d+)\)', filename).group(1))
        # small utility to find filename
        def get_filename(patientid, format):
            filename = 'COVID-19({patientid}).{format}'.format(patientid = patientid, format = format.lower())
            if (Path(self.path) / 'COVID-19' / filename).exists():
                return filename
            return 'COVID-19 ({patientid}).{format}'.format(patientid = patientid, format = format.lower())

        covid19_xray_samples['filename'] = covid19_xray_samples.apply(lambda row: get_filename(row['patientid'], row['FORMAT']), axis=1)
        covid19_xray_samples = covid19_xray_samples[~covid19_xray_samples['patientid'].isin(self.discard_pids)]
        print('SIRM unique patients: ', len(covid19_xray_samples['patientid'].unique()), ', total samples: ', len(covid19_xray_samples))

        for _, sample in covid19_xray_samples.iterrows():
            pid, filename = sample['patientid'], sample['filename']
            new_name = 'sirm_%s_%s' % (pid, filename)
            shutil.copy(str(Path(self.path) / 'COVID-19' / filename), str(covid19_dir / new_name))


class RSNA:
    def __init__(self, path):
        self.path = path
        self.normal_metadata = pd.read_csv(str(Path(path) / 'stage_2_detailed_class_info.csv'))

    def create_dataset(self, savepath):
        normal_dir = Path(savepath) / 'normal'
        ensure_exists(normal_dir)  # ensure path exists.

        normal_xray_samples = self.normal_metadata[self.normal_metadata['class']=='Normal'][['patientId']]
        normal_xray_samples['filename'] = normal_xray_samples['patientId'] + '.dcm'
        print('RSNA unique patients: ', len(normal_xray_samples['patientId'].unique()), ', total samples: ', len(normal_xray_samples))

        for _, sample in normal_xray_samples.iterrows():
            pid, filename = sample['patientId'], sample['filename']
            new_name = 'rsna_%s_%s' % (pid, pid + '.png')
            if not (normal_dir / new_name).exists():
                ds = dicom.dcmread(str(Path(self.path) / 'stage_2_train_images' / filename))
                pixel_array_numpy = ds.pixel_array

                png.from_array(pixel_array_numpy, 'L').save(str(normal_dir / new_name))


class PaulMooney:
    def __init__(self, path):
        self.path = path

    def create_dataset(self, savepath):
        normal_dir = Path(savepath) / 'normal'
        ensure_exists(normal_dir)  # ensure path exists.

        bacterial_dir = Path(savepath) / 'bacterial_pneumonia'
        ensure_exists(bacterial_dir)  # ensure path exists.

        viral_dir = Path(savepath) / 'viral_pneumonia'
        ensure_exists(viral_dir)  # ensure path exists.

        patient_ids = { 'normal': set(), 'virus': set(), 'bacteria': set() }
        for phase in ['train', 'val', 'test']:
            pdb.set_trace()
            for type in ['NORMAL', 'PNEUMONIA']:
                pdb.set_trace()
                for image_path in (Path(self.path) / phase / type).rglob('*.jpeg'):
                    filename = image_path.name
                    if type == 'NORMAL':
                        pid = filename.split('-', 2)[1]
                        new_name = 'paulmooney_%s_%s' % (pid, filename)
                        shutil.copy(str(Path(self.path) / phase / type / filename), str(normal_dir / new_name))
                        patient_ids['normal'].add(pid)
                    else:
                        pid, pneumonia_type, _ = filename.split('_', 2)
                        pneumonia_dir = bacterial_dir if 'bacteria' in pneumonia_type else viral_dir
                        new_name = 'paulmooney_%s_%s' % (pid, filename)
                        shutil.copy(str(Path(self.path) / phase / type / filename), str(pneumonia_dir / new_name))
                        patient_ids[pneumonia_type].add(pid)
                        
        print('Paul Mooney unique patients: ', {key: len(vals) for key, vals in patient_ids.items()})


if __name__ == '__main__':
    # create cohen dataset
    root_path = '/home/vshshv3/Downloads/covid19_xray/'

    cohen = JCohen(root_path + 'cohen')
    cohen_urls = cohen.create_dataset('/home/vshshv3/Downloads/covid19_xray_processed', only_urls = True)

    agchung = AGChung(root_path + 'agchung')
    agchung.create_dataset('/home/vshshv3/Downloads/covid19_xray_processed')

    actualmed = ActualMed(root_path + 'actualmed')
    actualmed.create_dataset('/home/vshshv3/Downloads/covid19_xray_processed')

    sirm = SIRM(root_path + 'rahman', cohen_urls)
    sirm.create_dataset('/home/vshshv3/Downloads/covid19_xray_processed')

    rsna = RSNA(root_path + 'rsna')
    rsna.create_dataset('/home/vshshv3/Downloads/covid19_xray_processed')

    paul_mooney = PaulMooney(root_path + 'paul_mooney')
    paul_mooney.create_dataset('/home/vshshv3/Downloads/covid19_xray_processed')
    exit(0)