"""Normalize the bounded public feasibility sample; no model/network calls."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/research/production_eval/execution'


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def utc_timestamp(value):
    if value is None:
        return None
    if isinstance(value,(int,float)):
        return datetime.fromtimestamp(value,tz=timezone.utc)
    result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=OUT/'raw_sample.jsonl')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw_bytes=args.input.read_bytes()
    raw=[json.loads(line) for line in raw_bytes.decode().splitlines()]
    normalized=[]
    counts=Counter()
    months=Counter()
    languages=Counter()
    seen=Counter()
    lag_days=[]
    for item in raw:
        row=item['row']; conv=row.get('conversation',[])
        first=conv[0] if conv else {}
        assistant=next((m for m in conv[1:] if m.get('role')=='assistant'),{})
        prompt=first.get('content','')
        reasons=[]
        if first.get('role')!='user': reasons.append('first_turn_is_not_user')
        if not isinstance(prompt,str): prompt=''; reasons.append('non_text_input')
        if not prompt.strip(): reasons.append('empty_input')
        if len(prompt)>12000: reasons.append('over_12000_character_cohort_limit')
        if item.get('truncated_cells'): reasons.append('server_truncated_record')
        if not assistant: reasons.append('missing_first_assistant')
        timestamp=None
        timestamp_source=None
        try:
            if first.get('timestamp') is not None:
                timestamp=utc_timestamp(first['timestamp']);timestamp_source='first_user_timestamp'
            elif assistant.get('timestamp') is not None:
                timestamp=utc_timestamp(assistant['timestamp']);timestamp_source='first_assistant_receipt_proxy'
            if timestamp is None: reasons.append('missing_request_time')
        except (ValueError,TypeError,OverflowError): reasons.append('invalid_request_time')
        h=digest(prompt)
        family=digest(prompt.strip())
        turn=first.get('turn_identifier')
        # Each sampled source row has a unique record identity; duplicate
        # requests remain linked through duplicate_family_id rather than
        # silently overwriting one another in annotation/review mappings.
        identifier=digest(item['dataset_revision']+'|'+str(item['row_idx'])+'|'+str(turn)+'|'+h)
        if seen[family]: reasons.append('duplicate_first_request_in_sample')
        seen[family]+=1
        role_number=int(family[:8],16)%10
        prospective_role='reference' if role_number<3 else ('development' if role_number==3 else 'construction')
        # Entire feasibility sample is development material once inspected.
        # The prospective role is only an audit field; these IDs cannot be used
        # as an untouched reference bank for a later confirmatory analysis.
        record={
            'example_id':identifier,'input_text':prompt,'input_hash':h,
            'duplicate_family_id':family,'source_turn_identifier':turn,
            'conversation_group_id':str(row.get('conversation_hash')),
            'dataset_revision':item['dataset_revision'],'source_row_index':item['row_idx'],
            'source_inclusion_probability':item['sampling']['inclusion_probability'],
            'source_design_weight':item['sampling']['design_weight'],
            'request_time':timestamp.isoformat() if timestamp else None,
            'request_time_source':timestamp_source,
            'naive_source_times_assumed_utc':True,
            'source_language':row.get('language'),
            'source_redacted':row.get('redacted'),
            'source_toxic':row.get('toxic'),
            'input_characters':len(prompt),
            'technical_eligible':not reasons,'technical_exclusion_reasons':reasons,
            'data_role':'feasibility_development',
            'prospective_hash_role_for_design_audit':prospective_role,
            'annotation_status':'not_human_reviewed',
        }
        normalized.append(record)
        counts['total']+=1;counts['technical_eligible']+=not reasons
        for reason in reasons: counts[reason]+=1
        languages[str(row.get('language'))]+=1
        if timestamp:
            months[timestamp.strftime('%Y-%m')]+=1
            try:
                last=utc_timestamp(row.get('timestamp'))
                if last:lag_days.append((last-timestamp).total_seconds()/86400)
            except (ValueError,TypeError):pass
    for name,rows in [('normalized_requests.jsonl',normalized),('audit_requests.jsonl',[r for r in normalized if r['technical_eligible']])]:
        path=args.output_dir/name
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={
        'status':'normalized_feasibility_sample','counts':dict(counts),
        'first_response_month_sample_counts':dict(sorted(months.items())),
        'source_language_sample_counts':dict(languages.most_common()),
        'final_minus_first_timestamp_over_one_day':sum(x>1 for x in lag_days),
        'max_final_minus_first_timestamp_days':max(lag_days,default=None),
        'population_monthly_counts':None,
        'human_review_completed':False,
        'later_confirmatory_reference_exclusion':'Every inspected feasibility example and its duplicate family must be excluded from untouched future reference roles.',
        'normalization_version':'first_user_v1',
        'input_length_cap_characters':12000,
        'raw_sample_sha256':hashlib.sha256(raw_bytes).hexdigest()
    }
    (args.output_dir/'normalization_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
