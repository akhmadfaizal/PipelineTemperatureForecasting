import sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'fungsi_bersama'))
import common

class DataIdentityTests(unittest.TestCase):
    def test_missing_columns_rejected_before_preparation(self):
        cfg=common.load_config(ROOT/'config.yaml')
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'changed.csv'
            source.write_text('waktu_wib,temperature\n2026-08-20,26\n')
            cfg['data']['csv_path']=str(source)
            with self.assertRaisesRegex(ValueError,'Kolom dataset belum lengkap'):
                common.prepare_dataset(cfg,Path(temp)/'run')
    def test_existing_run_cannot_be_overwritten(self):
        cfg=common.load_config(ROOT/'config.yaml')
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'sensor_raw.csv'
            source.write_text('waktu_wib,temperature,humidity\n2026-08-20,26,48\n')
            cfg['data']['csv_path'] = str(source)
            run = Path(temp) / 'run'
            (run / 'dataset').mkdir(parents=True)
            marker=run/'dataset/metadata.json';marker.write_text('{"original":true}')
            with self.assertRaises(FileExistsError):
                common.prepare_dataset(cfg,run)
            self.assertEqual(marker.read_text(),'{"original":true}')

if __name__=='__main__':unittest.main()
