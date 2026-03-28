from pgmpy.datasets import list_datasets
from pgmpy.example_models import list_models

print("=" * 70)
print("SECTION 1: list_models — backward compatibility (exact match)")
print("=" * 70)

all_models = list_models()
assert len(all_models) == 257, f"Expected 257 got {len(all_models)}"
print(f"PASS total models: {len(all_models)}")

r = list_models(is_parameterized=True)
assert len(r) > 0
assert all(isinstance(m, str) for m in r)
print(f"PASS is_parameterized=True: {len(r)} models")

r = list_models(is_discrete=True)
assert "bnlearn/alarm" in r
assert "bnlearn/arth150" not in r
print(f"PASS is_discrete=True: {len(r)} models")

r = list_models(is_continuous=True)
assert "bnlearn/arth150" in r
assert "bnlearn/alarm" not in r
print(f"PASS is_continuous=True: {len(r)} models")

r = list_models(name="bnlearn/alarm")
assert r == ["bnlearn/alarm"], f"Got {r}"
print(f"PASS exact name filter: {r}")

assert list_models() == sorted(list_models()), "Output not sorted"
print("PASS output is sorted")

print()
print("=" * 70)
print("SECTION 2: list_models — comparator suffixes")
print("=" * 70)

r_gt = list_models(n_nodes__gt=10)
assert len(r_gt) > 0
assert all(isinstance(m, str) for m in r_gt)
print(f"PASS n_nodes__gt=10: {len(r_gt)} models")

r_gte = list_models(n_nodes__gte=10)
assert len(r_gte) >= len(r_gt)
print(f"PASS n_nodes__gte=10: {len(r_gte)} models (>= gt)")

r_lt = list_models(n_nodes__lt=10)
assert len(r_lt) > 0
print(f"PASS n_nodes__lt=10: {len(r_lt)} models")

r_lte = list_models(n_nodes__lte=10)
assert len(r_lte) >= len(r_lt)
print(f"PASS n_nodes__lte=10: {len(r_lte)} models (>= lt)")

r_ne = list_models(n_nodes__ne=10)
r_eq = list_models(n_nodes=10)
assert len(r_ne) + len(r_eq) == len(all_models), f"{len(r_ne)} + {len(r_eq)} != {len(all_models)}"
print(f"PASS n_nodes__ne=10: {len(r_ne)} + exact {len(r_eq)} = {len(all_models)} total")

r_in = list_models(n_nodes__in=[10, 20, 46])
assert len(r_in) > 0
print(f"PASS n_nodes__in=[10,20,46]: {len(r_in)} models: {r_in[:3]}...")

print()
print("=" * 70)
print("SECTION 3: list_models — range queries")
print("=" * 70)

r_range = list_models(n_nodes__gte=10, n_nodes__lte=50)
assert len(r_range) > 0
assert len(r_range) <= len(all_models)
assert set(r_range).issubset(set(r_gte))
print(f"PASS n_nodes__gte=10, lte=50: {len(r_range)} models")

r_tight = list_models(n_nodes__gt=10, n_nodes__lt=20)
assert len(r_tight) <= len(r_range)
print(f"PASS n_nodes__gt=10, lt=20: {len(r_tight)} models")

r_impossible = list_models(n_nodes__gt=50, n_nodes__lt=10)
assert r_impossible == [], f"Expected [] got {r_impossible}"
print("PASS impossible range returns empty list")

print()
print("=" * 70)
print("SECTION 4: list_models — mixed exact + comparator")
print("=" * 70)

r_mixed = list_models(is_discrete=True, n_nodes__gt=10)
assert len(r_mixed) > 0
assert set(r_mixed).issubset(set(list_models(is_discrete=True)))
assert set(r_mixed).issubset(set(r_gt))
print(f"PASS is_discrete=True + n_nodes__gt=10: {len(r_mixed)} models")

r_mixed2 = list_models(is_parameterized=True, n_nodes__gte=10, n_nodes__lte=50)
assert len(r_mixed2) > 0
print(f"PASS is_parameterized=True + range: {len(r_mixed2)} models")

print()
print("=" * 70)
print("SECTION 5: list_models — edge cases")
print("=" * 70)

r_empty = list_models(n_nodes__gt=999999)
assert r_empty == []
print("PASS n_nodes__gt=999999 returns empty list")

r_edges = list_models(n_edges__gt=10)
assert len(r_edges) > 0
print(f"PASS n_edges__gt=10: {len(r_edges)} models")

r_in_single = list_models(n_nodes__in=[8])
assert isinstance(r_in_single, list)
print(f"PASS n_nodes__in single value: {r_in_single}")

r_in_empty = list_models(n_nodes__in=[])
assert r_in_empty == []
print("PASS n_nodes__in=[] returns empty list")

print()
print("=" * 70)
print("SECTION 6: list_models — error handling")
print("=" * 70)

try:
    list_models(is_paraterized=True)
    print("FAIL: should have raised ValueError")
except ValueError as e:
    assert "Unrecognized filter argument" in str(e)
    print("PASS typo tag raises ValueError")

try:
    list_models(num_nodes=10)
    print("FAIL: should have raised ValueError")
except ValueError as e:
    assert "Unrecognized filter argument" in str(e)
    print("PASS wrong key raises ValueError")

try:
    list_models(num_nodes__gt=10)
    print("FAIL: should have raised ValueError")
except ValueError as e:
    assert "Unrecognized filter argument" in str(e)
    print("PASS wrong key with suffix raises ValueError")

try:
    list_models(n_nodes__in=10)
    print("FAIL: should have raised TypeError")
except TypeError as e:
    assert "list, tuple, or set" in str(e)
    print("PASS __in with scalar raises TypeError")

try:
    list_models(n_nodes__in="abc")
    print("FAIL: should have raised TypeError")
except TypeError as e:
    assert "list, tuple, or set" in str(e)
    print("PASS __in with string raises TypeError")

print()
print("=" * 70)
print("SECTION 7: list_datasets — backward compatibility")
print("=" * 70)

all_ds = list_datasets()
assert len(all_ds) > 0
print(f"PASS total datasets: {len(all_ds)}")

r = list_datasets(is_continuous=True)
assert "abalone_continuous" in r
assert "sachs_discrete" not in r
assert "abalone_mixed" not in r
print(f"PASS is_continuous=True: {len(r)} datasets")

r = list_datasets(has_ground_truth=True)
assert "abalone_continuous" not in r
print(f"PASS has_ground_truth=True: {len(r)} datasets")

r = list_datasets(is_discrete=True, has_ground_truth=True)
assert "sachs_discrete" in r
print(f"PASS is_discrete + has_ground_truth: {r}")

assert list_datasets() == sorted(list_datasets())
print("PASS output is sorted")

print()
print("=" * 70)
print("SECTION 8: list_datasets — comparator suffixes")
print("=" * 70)

r_gt = list_datasets(n_samples__gt=1000)
assert len(r_gt) > 0
assert all(isinstance(d, str) for d in r_gt)
print(f"PASS n_samples__gt=1000: {len(r_gt)} datasets")

r_gte = list_datasets(n_samples__gte=1000)
assert len(r_gte) >= len(r_gt)
print(f"PASS n_samples__gte=1000: {len(r_gte)} datasets")

r_lt = list_datasets(n_samples__lt=1000)
assert len(r_lt) > 0
print(f"PASS n_samples__lt=1000: {len(r_lt)} datasets")

r_lte = list_datasets(n_samples__lte=1000)
assert len(r_lte) >= len(r_lt)
print(f"PASS n_samples__lte=1000: {len(r_lte)} datasets")

r_var_gt = list_datasets(n_variables__gt=5)
assert len(r_var_gt) > 0
print(f"PASS n_variables__gt=5: {len(r_var_gt)} datasets")

r_var_range = list_datasets(n_variables__gte=5, n_variables__lte=20)
assert len(r_var_range) > 0
assert set(r_var_range).issubset(set(r_var_gt) | set(list_datasets(n_variables__gte=5)))
print(f"PASS n_variables range: {len(r_var_range)} datasets")

r_in = list_datasets(n_variables__in=[5, 10, 15])
assert isinstance(r_in, list)
print(f"PASS n_variables__in=[5,10,15]: {len(r_in)} datasets")

print()
print("=" * 70)
print("SECTION 9: list_datasets — mixed exact + comparator")
print("=" * 70)

r_mixed = list_datasets(is_continuous=True, n_samples__gt=500)
assert len(r_mixed) > 0
assert set(r_mixed).issubset(set(list_datasets(is_continuous=True)))
print(f"PASS is_continuous + n_samples__gt=500: {len(r_mixed)} datasets")

r_impossible = list_datasets(n_samples__gt=999999)
assert r_impossible == []
print("PASS n_samples__gt=999999 returns empty list")

print()
print("=" * 70)
print("SECTION 10: list_datasets — error handling")
print("=" * 70)

try:
    list_datasets(is_paraterized=True)
    print("FAIL: should have raised ValueError")
except ValueError as e:
    assert "Unrecognized filter argument" in str(e)
    print("PASS typo tag raises ValueError")

try:
    list_datasets(num_samples=100)
    print("FAIL: should have raised ValueError")
except ValueError as e:
    assert "Unrecognized filter argument" in str(e)
    print("PASS wrong key raises ValueError")

try:
    list_datasets(n_samples__in=100)
    print("FAIL: should have raised TypeError")
except TypeError as e:
    assert "list, tuple, or set" in str(e)
    print("PASS __in with scalar raises TypeError")

print()
print("=" * 70)
print("SECTION 11: consistency checks")
print("=" * 70)

gt10 = set(list_models(n_nodes__gt=10))
lte10 = set(list_models(n_nodes__lte=10))
eq10 = set(list_models(n_nodes=10))
all_set = set(all_models)
assert gt10.isdisjoint(lte10), "gt and lte must not overlap"
assert eq10.issubset(lte10), "exact=10 must be subset of lte=10"
assert gt10 | lte10 == all_set, "gt10 + lte10 must cover all models"
print("PASS gt/lte partition covers all models with no overlap")

gte10 = set(list_models(n_nodes__gte=10))
lt10 = set(list_models(n_nodes__lt=10))
assert gte10.isdisjoint(lt10), "gte and lt must not overlap"
assert gte10 | lt10 == all_set, "gte10 + lt10 must cover all models"
print("PASS gte/lt partition covers all models with no overlap")

ne10 = set(list_models(n_nodes__ne=10))
assert ne10 | eq10 == all_set
assert ne10.isdisjoint(eq10)
print("PASS ne + eq covers all models with no overlap")

print()
print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
