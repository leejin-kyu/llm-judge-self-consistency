# -*- coding: utf-8 -*-
"""P5-33: Export the exact prompt templates to prompts/ for release."""
import os
import judges

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "prompts")
os.makedirs(DST, exist_ok=True)

FILES = {
    "direct_system.txt": judges.PAIRWISE_SYS,
    "direct_user_template.txt": judges.PAIRWISE_USER,
    "cot_system.txt": judges.COT_SYS,
    "cot_user_template.txt": judges.COT_USER,
    "criterion_cot_system.txt": judges.COT_CRIT_SYS,
    "criteria.txt": "\n".join(f"{k}: {v}" for k, v in judges.CRITERIA.items()),
}
for name, content in FILES.items():
    open(os.path.join(DST, name), "w", encoding="utf-8").write(content + "\n")
print(f"wrote {len(FILES)} prompt files to {DST}")
