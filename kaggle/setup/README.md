# Offline vLLM wheelhouse

`build_quant_wheels.py` tạo một wheelhouse generic cho vLLM và toàn bộ dependency
dạng wheel. Chạy file này trong Kaggle notebook có **Internet ON**, dùng cùng image
và Python version với notebook GPU sẽ chạy offline.

Quy trình:

1. Copy `build_quant_wheels.py` vào notebook Internet ON và Run All.
2. Save `/kaggle/working/vllm_wheels` thành một Kaggle Dataset.
3. Add Dataset đó vào notebook `experiments/baseline.py` có Internet OFF.
4. Điền đúng path Dataset vào `VLLM_WHEELS_DIR` trong notebook baseline. Notebook
   kiểm tra `manifest.json` và SHA-256 của từng wheel trước khi cài offline.

Script cố ý không chọn “latest”. Phải ghim exact version đã audit trước khi chạy:

```python
import os
os.environ["VLLM_SPEC"] = "vllm==<audited-version>"
```

Dependency thêm, nếu checkpoint thực sự cần, được truyền bằng danh sách phân cách
bởi dấu phẩy:

```python
os.environ["VLLM_EXTRA_SPECS"] = "bitsandbytes==<audited-version>"
```

Không dùng wheelhouse được tạo từ Python/CUDA image khác mà không kiểm tra lại
compatibility.

`manifest.json` ghi exact requirement, tên file, kích thước và SHA-256. Không sửa
wheelhouse sau khi tạo; nếu cần đổi package/version, tạo Dataset mới.
Script sẽ dừng nếu thư mục output đã có file, để wheel của hai lần resolve khác
nhau không bị trộn trong cùng một manifest.
