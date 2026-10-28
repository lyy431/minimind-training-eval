"""Correct direction labels for a lower-is-better metric, without changing numbers."""
import shutil
import chat_repair_common as e

def main():
    s=e.read(e.OUT/'summary.json');paired=s['retention']['nll296']['paired']
    if paired.get('lower_is_better'):return
    shutil.copy2(e.OUT/'summary.json',e.OUT/'summary_before_direction_labels.json')
    positive,negative=paired['better'],paired['worse']
    paired.update(better=negative,worse=positive,positive_differences=positive,negative_differences=negative,lower_is_better=True)
    e.save('summary.json',s)
    e.save('metric_direction_annotation.json',dict(changed='NLL better/worse labels only; lower is better',
        numerical_differences_standard_errors_intervals_and_acceptance_unchanged=True,
        loss_increased=positive,loss_decreased=negative,source_sha256=e.sha(__file__)))
    print('NLL direction labels clarified; all acceptance and numerical statistics unchanged.',flush=True)

if __name__=='__main__':main()
