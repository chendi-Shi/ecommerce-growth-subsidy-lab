"""Download at runtime; raw third-party records are not redistributed."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCE = 'https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz'
PRIMARY = 'http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv'
SOURCE_MD5 = 'a68a81291f53a14f4e29002629803ba3'
ARMS = {'No E-Mail':0,'Mens E-Mail':1,'Womens E-Mail':2}
FIELDS = ['customer_id','recency','history','mens','womens','newbie','zip_code',
          'channel','treatment','visit','conversion','spend']


def prepare(root=Path('data')):
    raw = root/'raw'/'hillstrom.csv.gz'
    raw.parent.mkdir(parents=True,exist_ok=True)
    if not raw.exists():
        request = urllib.request.Request(SOURCE,headers={'User-Agent':'ecommerce-growth-research/2.0'})
        with urllib.request.urlopen(request,timeout=60) as response:
            payload = response.read()
        if hashlib.md5(payload).hexdigest() != SOURCE_MD5:
            raise ValueError('Source checksum differs: stop and inspect the dataset version.')
        raw.write_bytes(payload)
    payload=raw.read_bytes()
    if hashlib.md5(payload).hexdigest() != SOURCE_MD5:
        raise ValueError('Cached dataset checksum mismatch.')
    destination=root/'processed'/'hillstrom.csv'
    destination.parent.mkdir(parents=True,exist_ok=True)
    counts={0:0,1:0,2:0}
    with gzip.open(raw,'rt',encoding='utf-8-sig',newline='') as source, destination.open('w',encoding='utf-8',newline='') as target:
        reader=csv.DictReader(source)
        required={'recency','history','mens','womens','newbie','zip_code','channel','segment','visit','conversion','spend'}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f'Unexpected schema: {reader.fieldnames}')
        writer=csv.DictWriter(target,fieldnames=FIELDS); writer.writeheader()
        for i,row in enumerate(reader):
            out={name:row[name] for name in FIELDS if name not in ['customer_id','treatment']}
            out['customer_id']=f'h{i:08d}'  # stable row key, not a disclosed original user ID
            out['treatment']=ARMS[row['segment']]
            counts[out['treatment']]+=1; writer.writerow(out)
    if sum(counts.values())!=64000:
        raise ValueError(f'Expected entire 64000-recipient release, got {counts}.')
    manifest=dict(dataset='Hillstrom 2008 retail email randomized experiment',rows=64000,
        source_url=SOURCE,original_csv_url=PRIMARY,
        primary_description='https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html',
        md5=SOURCE_MD5,sha256=hashlib.sha256(payload).hexdigest(),
        arm_counts=counts,assignment_probabilities=[1/3]*3,outcome_window_days=14,currency='USD',
        customer_id='Stable row index assigned locally; source does not disclose customer IDs.',
        population='Customers who last purchased within prior 12 months; existing-customer activation.',
        redistribution='No explicit original dataset licence found; runtime download only, no raw row upload.')
    (destination.parent/'hillstrom_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return destination,manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,default=Path('data'))
    args=parser.parse_args(); print(json.dumps(prepare(args.data_root)[1],indent=2))
