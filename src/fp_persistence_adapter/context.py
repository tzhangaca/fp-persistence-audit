"""Pre-audit window/exposure/context adapter. No predictive modeling or outcome optimization.

Use with the generic fp_persistence.audit(..., final_stratum_col='final_stratum')
after the user freezes window and context definition. Input rows MUST already be
restricted to the legitimate audit population when building context strata.
"""
import numpy as np
import pandas as pd


def exact_legitimate_budget(n):
    """Return the frozen paper-profile 2% count without floating arithmetic."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError('n must be an integer')
    if int(n) < 0:
        raise ValueError('n must be nonnegative')
    return (int(n) + 25) // 50


def exposure_table(legitimate, entity_col='entity_id'):
    if not isinstance(legitimate, pd.DataFrame):
        raise ValueError('legitimate must be a pandas DataFrame')
    if entity_col not in legitimate.columns:
        raise ValueError(f'missing entity column: {entity_col}')
    if legitimate[entity_col].isna().any():
        raise ValueError('null entity ID')
    out = legitimate.groupby(entity_col, sort=False, observed=True).size().rename('legitimate_exposure')
    return out.reset_index()


def _aware_utc(values, name):
    """Parse explicitly timezone-aware values and normalize them to UTC."""
    for value in values.tolist():
        if pd.isna(value):
            raise ValueError(f'{name} must not contain missing timestamps')
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            raise ValueError(f'{name} must contain explicit timezone-aware timestamps')
    return pd.to_datetime(values, utc=True, errors='raise')


def _require_columns(df, columns):
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f'missing columns: {missing}')


def _reject_stringification_collisions(df, columns):
    """Prevent distinct typed context values from collapsing to one label."""
    for name in columns:
        mapping = {}
        for value in df[name].tolist():
            rendered = str(value)
            typed = (type(value).__module__, type(value).__qualname__, repr(value))
            prior = mapping.setdefault(rendered, typed)
            if prior != typed:
                raise ValueError(f'ambiguous string representation in context column: {name}')


def time_and_amount_context(legitimate_audit, train_legitimate_amounts,
                            timestamp_col='timestamp', amount_col='amount',
                            coarse_col='merchant_category'):
    """Return context table and four training-defined amount quintile cutpoints.

    Timestamp should already have a fixed UTC policy; no timezone inference.
    """
    if not isinstance(legitimate_audit, pd.DataFrame):
        raise ValueError('legitimate_audit must be a pandas DataFrame')
    _require_columns(
        legitimate_audit,
        [timestamp_col, amount_col] + ([coarse_col] if coarse_col is not None else []),
    )
    if legitimate_audit.empty:
        raise ValueError('empty legitimate audit')
    x = pd.to_numeric(pd.Series(train_legitimate_amounts), errors='raise')
    if x.empty:
        raise ValueError('empty train amounts')
    if x.isna().any() or not np.isfinite(x).all():
        raise ValueError('bad train amounts')
    cuts = x.quantile([.2,.4,.6,.8], interpolation='linear').to_numpy()
    a = legitimate_audit.copy()
    ts = _aware_utc(a[timestamp_col], timestamp_col)
    if ts.isna().any():
        raise ValueError('null timestamp')
    amt = pd.to_numeric(a[amount_col], errors='raise')
    if amt.isna().any() or not np.isfinite(amt).all():
        raise ValueError('bad audit amount')
    a['audit_day'] = ts.dt.strftime('%Y-%m-%d')
    a['six_hour_block'] = (ts.dt.hour // 6).astype(np.int8)
    # searchsorted(right) resolves ties at cutpoints deterministically
    a['amount_bin'] = np.searchsorted(cuts, amt.to_numpy(), side='right').astype(np.int8)
    if coarse_col is not None:
        nonmissing = a.loc[a[coarse_col].notna(), [coarse_col]]
        _reject_stringification_collisions(nonmissing, [coarse_col])
        if nonmissing[coarse_col].astype(str).eq('__MISSING__').any():
            raise ValueError('reserved coarse-context value: __MISSING__')
        a['coarse_context'] = a[coarse_col].fillna('__MISSING__').astype(str)
    else:
        a['coarse_context'] = '__NO_COARSE_CONTEXT__'
    return a, cuts.tolist()


def resolve_residual_strata(df, day='audit_day', block='six_hour_block',
                            amount='amount_bin', coarse='coarse_context', minimum=20):
    """Freeze a disjoint fine-to-coarse residual partition; no global regrouping.

    All input rows must be legitimate; no flags or scores are read here.
    Assignment levels: (day,block,amount,coarse), (day,block,amount),
    (day,block), then day. Accept ONLY groups >=minimum on *remaining rows*.
    Any residual final day below minimum is an error rather than an implicit
    undocumented fallback. Returns a Series aligned to df.index; unique index
    and no null context keys are required.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be a pandas DataFrame')
    if df.empty:
        raise ValueError('empty context table')
    if df.index.has_duplicates:
        raise ValueError('duplicate index')
    if isinstance(minimum, (bool, np.bool_)) or not isinstance(minimum, (int, np.integer)) or minimum < 1:
        raise ValueError('minimum must be a positive integer')
    keys = [day,block,amount]+([coarse] if coarse is not None else [])
    _require_columns(df, keys)
    if df[keys].isna().any().any():
        raise ValueError('missing context fields')
    _reject_stringification_collisions(df, keys)
    assigned = pd.Series(pd.NA, index=df.index, dtype='string')
    level_keys = ([(day,block,amount,coarse)] if coarse is not None else []) + [(day,block,amount),(day,block),(day,)]
    for level, cols in enumerate(level_keys):
        unresolved = assigned.isna()
        if not unresolved.any(): break
        subset = df.loc[unresolved, list(cols)]
        # groupby .transform size computed only among unresolved records
        sizes = subset.groupby(list(cols), sort=False, observed=True)[cols[0]].transform('size')
        accept = sizes.ge(minimum)
        if level==len(level_keys)-1 and not bool(accept.all()):
            raise ValueError('unresolved final day below minimum; cannot silently alter null')
        if not accept.any(): continue
        rows = subset.loc[accept, list(cols)]
        labels = rows.astype(str).agg('|'.join,axis=1)
        # The public audit groups on the returned label. Reject ambiguous label
        # encodings instead of silently merging distinct typed context tuples.
        tuple_by_label = {}
        for label, values in zip(labels.tolist(), rows.itertuples(index=False, name=None)):
            typed = tuple((type(v).__module__, type(v).__qualname__, repr(v)) for v in values)
            prior = tuple_by_label.setdefault(label, typed)
            if prior != typed:
                raise ValueError('ambiguous composite final-stratum label')
        assigned.loc[rows.index] = f'L{level}|' + labels
    if assigned.isna().any(): raise AssertionError('unassigned rows')
    sizes = assigned.value_counts()
    if sizes.min() < minimum: raise AssertionError('undersized stratum')
    return assigned.rename('final_stratum')
