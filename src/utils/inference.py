import torch
from torch.quantization import quantize_dynamic

def quantize_model(model, example_input, dtype='fp16', backend='fbgemm'):
    quantized_model = model
    
    if dtype == 'int8':
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )
    elif dtype == 'fp16':
        if torch.cuda.is_available():
            quantized_model = model.half().cuda()
            example_input = example_input.half().cuda()
        else:
            quantized_model = model.half()
            example_input = example_input.half()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    quantized_model.eval()
    with torch.no_grad():
        quantized_model(example_input)
    
    return quantized_model


def inference_speed(model, input_tensor, num_runs=100, warmup=10):
    model.eval()
    times = []
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    return sum(times) / len(times) * 1000
