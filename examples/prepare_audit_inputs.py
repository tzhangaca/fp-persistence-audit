"""Generic pre-audit adapter example, for use alongside (not inside) fp-persistence-audit v0.1.2.

Requires a user-selected and documented period, legitimate labels, and training
period. Never silently defines an analysis window or chooses model features.
This script outputs legitimate audit rows with resolved final_stratum plus a
(nonidentifiable) structured report; user IDs are only written with --write-events.
"""
import argparse, hashlib, json, pathlib
import numpy as np
import pandas as pd
from fp_persistence_adapter import time_and_amount_context, resolve_residual_strata, exposure_table, exact_legitimate_budget


def _parse_calendar_date(value, name):
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f'invalid {name}: {value}') from exc
    if parsed.tzinfo is not None:
        raise ValueError(f'{name} must be an ISO calendar date without a timezone')
    if parsed != parsed.normalize():
        raise ValueError(f'{name} must not include a time of day')
    return parsed


def _strict_binary(series, name):
    values = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(int(value))
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and int(value) in (0, 1):
            values.append(int(value))
        else:
            raise ValueError(f'{name} must contain only strict binary values')
    return pd.Series(values, index=series.index, dtype='int8')


def _aware_utc(series, name):
    for value in series.tolist():
        if pd.isna(value):
            raise ValueError(f'{name} must not contain missing timestamps')
        if pd.Timestamp(value).tzinfo is None:
            raise ValueError(f'{name} must be explicitly timezone-aware')
    return pd.to_datetime(series, utc=True, errors='raise')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='input file; assumes CSV')
    p.add_argument('--train-start',required=True);p.add_argument('--train-end',required=True)
    p.add_argument('--audit-start',required=True);p.add_argument('--audit-end',required=True)
    p.add_argument('--timestamp',default='timestamp');p.add_argument('--event',default='transaction_id')
    p.add_argument('--entity',default='customer_id');p.add_argument('--legitimate-label',default='is_fraud')
    p.add_argument('--amount',default='amount');p.add_argument('--context',default='merchant_category')
    p.add_argument('--minimum',type=int,default=20);p.add_argument('--exposure-min',type=int,default=5)
    p.add_argument('--exposure-max',type=int,default=19);p.add_argument('--out',required=True)
    p.add_argument('--write-events',action='store_true',help='exports rows with IDs; keep private')
    args=p.parse_args()
    train_start = _parse_calendar_date(args.train_start, 'train-start')
    train_end = _parse_calendar_date(args.train_end, 'train-end')
    audit_start = _parse_calendar_date(args.audit_start, 'audit-start')
    audit_end = _parse_calendar_date(args.audit_end, 'audit-end')
    if train_start > train_end:
        raise ValueError('train-start must not be after train-end')
    if audit_start > audit_end:
        raise ValueError('audit-start must not be after audit-end')
    if train_end >= audit_start:
        raise ValueError('train/audit overlap')
    if args.minimum < 1:
        raise ValueError('minimum must be >=1')
    if args.exposure_min < 0 or args.exposure_max < args.exposure_min:
        raise ValueError('invalid exposure bounds')
    df=pd.read_csv(args.source,low_memory=False)
    required = [args.timestamp, args.event, args.entity, args.legitimate_label, args.amount, args.context]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f'missing columns: {missing}')
    ts=_aware_utc(df[args.timestamp], args.timestamp)
    day=ts.dt.strftime('%Y-%m-%d')
    labels = _strict_binary(df[args.legitimate_label], args.legitimate_label)
    train=df.loc[day.between(args.train_start,args.train_end) & (labels==0)]
    audit=df.loc[day.between(args.audit_start,args.audit_end) & (labels==0)].copy()
    if train.empty or audit.empty:raise ValueError('no legitimate training or audit transactions')
    if audit[args.event].isna().any():raise ValueError('missing event ID')
    if audit[args.event].duplicated().any():raise ValueError('duplicate event ID')
    if audit[args.entity].isna().any():raise ValueError('missing entity ID')
    a,cuts=time_and_amount_context(audit,train[args.amount],timestamp_col=args.timestamp,
                                  amount_col=args.amount,coarse_col=args.context)
    a['final_stratum']=resolve_residual_strata(a,minimum=args.minimum)
    ex=exposure_table(a,entity_col=args.entity)
    # Audit library defines legitimate_col=1 for legitimate; source label is inverse.
    a['legitimate'] = 1
    sourcehash=_sha256(args.source)
    report={'input_source_name':pathlib.Path(args.source).name, 'source_sha256':sourcehash, 'selected_train':[args.train_start,args.train_end],
            'selected_audit':[args.audit_start,args.audit_end], 'amount_cutpoints':cuts,
            'eligible_entities':int(ex.legitimate_exposure.between(args.exposure_min,args.exposure_max).sum()),
            'total_legitimate_events':len(a),'exact_2pct_budget':exact_legitimate_budget(len(a)),
            'final_stratum_count':int(a.final_stratum.nunique()),
            'level_event_counts':a.final_stratum.str.split('|',regex=False).str[0].value_counts().to_dict(),
            'note':'not a prediction model, no FP flags generated, user determines operating point'}
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'audit_structure.json').write_text(json.dumps(report,indent=2))
    if args.write_events:
        a[[args.event,args.entity,'final_stratum',args.timestamp,'legitimate']].to_csv(out/'PRIVATE_legitimate_audit_with_context.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
