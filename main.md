# Local AI Models: Complete Mobile Deployment Guide
## iOS & Android using Unsloth, Llama.cpp (GGUF), and ExecuTorch (PTE)

**Target Framework:** Flutter

---

## Table of Contents

- [Part 1: Mobile Deployment Strategies](#part-1-mobile-deployment-strategies)
  - [Executive Summary](#executive-summary)
  - [Prerequisites](#prerequisites)
  - [Strategy A: GGUF with Llama.cpp](#strategy-a-gguf-with-llamacpp)
  - [Strategy B: PTE with ExecuTorch](#strategy-b-pte-with-executorch)
- [Part 2: Arny Mobile AI Architecture](#part-2-arny-mobile-ai-architecture)
  - [Architectural Strategies Overview](#architectural-strategies-overview)
  - [Plan A: Universal Monolith Implementation](#plan-a-universal-monolith-implementation)
  - [Plan B: Specialized Hybrid Implementation](#plan-b-specialized-hybrid-implementation)
- [Part 3: Performance Analysis & Recommendations](#part-3-performance-analysis--recommendations)

---

# Part 1: Mobile Deployment Strategies

## Executive Summary

This guide outlines two distinct technical strategies for deploying custom, fine-tuned Large Language Models (LLMs) directly onto mobile devices (iOS and Android). Both strategies utilize **Unsloth** for efficient fine-tuning but diverge significantly at the export and inference stages.

![Strategy Comparison](https://raw.githubusercontent.com/manumishra12/test/main/fig/strategy_comparison.png)
*Figure 1: Strategy Comparison - GGUF vs PTE Deployment Paths*

### **Strategy A (GGUF) - Recommended for Most Use Cases**

**Engine:** llama.cpp  
**Target Hardware:** CPU (with some GPU support)  
**File Format:** `.gguf` (single binary file)  
**Flutter Integration:** Direct via `llama_cpp_dart` package

**Advantages:**
- High stability and mature ecosystem
- Broad device support (runs on any CPU)
- Zero native code required (pure Dart)
- Cross-platform consistency
- Easy to debug and maintain
- Smaller development overhead

**Disadvantages:**
- Slower inference than hardware-accelerated solutions
- Higher battery consumption on CPU-intensive tasks
- Limited to CPU performance capabilities

**Best For:**
- Rapid prototyping and MVP development
- Applications where 2-3 second response time is acceptable
- Teams without mobile native development expertise
- Cross-platform deployment requirements

---

### **Strategy B (PTE) - For Performance-Critical Applications**

**Engine:** ExecuTorch (PyTorch Mobile Stack)  
**Target Hardware:** NPU/GPU/CPU with hardware acceleration  
**File Format:** `.pte` (PyTorch Edge format)  
**Flutter Integration:** Manual via MethodChannels (Swift/Kotlin)

**Advantages:**
- Hardware acceleration (NPU/GPU)
- Significantly faster inference (<1s responses)
- Lower battery consumption with NPU
- Official PyTorch mobile support
- Memory-mapped model loading

**Disadvantages:**
- Complex setup and integration
- Requires native code (Swift/Kotlin/C++)
- Platform-specific optimizations needed
- Steeper learning curve
- More debugging complexity

**Best For:**
- Production applications requiring instant responses
- Apps targeting premium devices with NPU support
- Teams with native mobile development expertise
- Performance-critical use cases

---

## Prerequisites

### Development Environment

| Requirement | Specification | Notes |
|-------------|---------------|-------|
| **Python** | 3.10 or higher | Use Conda for environment management |
| **Training Hardware** | NVIDIA GPU | Google Colab T4/A100 recommended |
| **Flutter SDK** | 3.22+ | Stable channel recommended |
| **Target Devices** | iOS 16+ / Android 10+ | Minimum 4GB RAM (8GB for 3B models) |
| **Storage** | 5-10 GB free | For model files and build artifacts |

### Software Dependencies

```bash
# Python packages
pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
pip install torch transformers datasets accelerate bitsandbytes

# System tools
- Node.js 16+ (Flutter tooling)
- Xcode 14+ (iOS development)
- Android Studio (Android development)
```

---

## Phase 1: Model Training (Common Core)

Both strategies begin with the same training process. We use **Llama-3.2-3B-Instruct** as the base model because it offers the optimal balance between model intelligence and mobile-friendly size.

![Training Architecture](https://raw.githubusercontent.com/manumishra12/test/main/fig/training_architecture.png)
*Figure 2: Model Training Pipeline - From Dataset to Deployment*

### Why Llama-3.2-3B-Instruct?

- **Size:** ~6GB full precision, ~2GB quantized
- **Performance:** Strong reasoning capabilities for mobile use cases
- **Instruction-tuned:** Pre-trained to follow instructions effectively
- **Mobile-optimized:** Designed with on-device inference in mind

### Step 1: Environment Setup

```bash
# Create Conda environment
conda create -n unsloth_mobile python=3.10 -y
conda activate unsloth_mobile

# Install dependencies
pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
pip install torch torchvision torchaudio
pip install transformers datasets accelerate bitsandbytes
```

### Step 2: Fine-Tuning Script

```python
# train_mobile_model.py
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Configuration
MODEL_NAME = "unsloth/llama-3.2-3b-instruct"
MAX_SEQ_LENGTH = 2048
LORA_R = 16

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)

# Train (add your dataset and trainer here)
# trainer.train()

# Save
model.save_pretrained_merged("llama_3_2_3b_mobile_merged", tokenizer)
```

**Key Concepts:**

**LoRA (Low-Rank Adaptation):**
- Parameter-efficient fine-tuning technique
- Updates small "adapter" matrices instead of all parameters
- Reduces training time and memory by 90%+

**4-bit Quantization:**
- Reduces model precision from 32-bit to 4-bit
- Cuts memory usage from ~12GB to ~2GB
- Minimal quality impact (<5% performance drop)

---

## Strategy A: GGUF with Llama.cpp

![GGUF Export Process](https://raw.githubusercontent.com/manumishra12/test/main/fig/gguf_export.png)
*Figure 3: GGUF Export & Deployment Flow*

### Export to GGUF

```python
# export_to_gguf.py
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama_3_2_3b_mobile_merged",
    max_seq_length=2048,
    load_in_4bit=False,
)

model.save_pretrained_gguf(
    "mobile_model",
    tokenizer,
    quantization_method="q4_k_m",  # Recommended for mobile
)
```

**Quantization Methods:**

| Method | Size (3B) | Speed | Quality | Use Case |
|--------|-----------|-------|---------|----------|
| `f16` | ~6 GB | Slow | Best | High-end devices only |
| `q8_0` | ~3 GB | Medium | Very Good | Balanced performance |
| `q4_k_m` | ~2 GB | Fastest | Acceptable | **Best for mobile** |

### Flutter Integration

**Project Structure:**
```
your_flutter_app/
├── lib/
│   ├── services/
│   │   └── ai_service_gguf.dart
│   └── screens/
│       └── chat_screen.dart
├── assets/
│   └── models/
│       └── model.gguf
└── pubspec.yaml
```

**pubspec.yaml:**
```yaml
dependencies:
  flutter:
    sdk: flutter
  llama_cpp_dart: ^1.0.0

flutter:
  assets:
    - assets/models/model.gguf
```

**AI Service:**
```dart
// lib/services/ai_service_gguf.dart
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class AiServiceGguf {
  LlamaProcessor? _processor;
  bool _isInitialized = false;
  
  Future<void> initialize() async {
    _processor = LlamaProcessor(
      'assets/models/model.gguf',
      nThreads: 4,
      nCtx: 2048,
      nBatch: 512,
    );
    await _processor!.initialize();
    _isInitialized = true;
  }
  
  Stream<String> generateStream({
    required String prompt,
    String? systemPrompt,
    int maxTokens = 512,
    double temperature = 0.7,
  }) async* {
    final formattedPrompt = _formatPrompt(prompt, systemPrompt);
    
    await for (final token in _processor!.stream(
      formattedPrompt,
      maxTokens: maxTokens,
      temperature: temperature,
      stop: ['<|eot_id|>', '<|end_of_text|>'],
    )) {
      yield token;
    }
  }
  
  String _formatPrompt(String userMessage, String? systemMessage) {
    return '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
${systemMessage ?? 'You are a helpful mobile AI assistant.'}<|eot_id|><|start_header_id|>user<|end_header_id|>
$userMessage<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''';
  }
  
  void dispose() {
    _processor?.dispose();
  }
}
```

---

## Strategy B: PTE with ExecuTorch

![PTE Export Process](https://raw.githubusercontent.com/manumishra12/test/main/fig/pte_export.png)
*Figure 4: PTE Export & Deployment Flow with Native Bridges*

### Phase 1: Environment Setup

```bash
# Create environment
conda create -n executorch_env python=3.10 -y
conda activate executorch_env

# Install dependencies
pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
pip install optimum[exporters]
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install executorch-sdk
```

### Phase 2: Export to PTE

```python
# export_to_pte.py
import subprocess

MODEL_PATH = "llama_3_2_3b_mobile_merged"
OUTPUT_DIR = "executorch_output"

cmd = [
    "optimum-cli", "export", "executorch",
    "--model", MODEL_PATH,
    "--output_dir", OUTPUT_DIR,
    "--task", "text-generation",
    "--recipe", "xnnpack",  # CPU-optimized
    "--quantization", "q4",
]

subprocess.run(cmd, check=True)
```

**Recipe Options:**

| Recipe | Platform | Hardware | Complexity | Performance |
|--------|----------|----------|------------|-------------|
| `xnnpack` | iOS + Android | CPU | Easy | Good |
| `metal` | iOS only | GPU | Hard | Excellent |
| `vulkan` | Android | GPU | Very Hard | Excellent |
| `qnn` | Qualcomm | NPU | Expert | Outstanding |

### Phase 3: Flutter Bridge

```dart
// lib/services/ai_service_pte.dart
import 'package:flutter/services.dart';

class AiServicePte {
  static const MethodChannel _channel = 
      MethodChannel('com.example.app/executorch');
  
  bool _isInitialized = false;
  
  Future<void> initialize({String modelPath = 'model.pte'}) async {
    final result = await _channel.invokeMethod<bool>(
      'initializeModel',
      {'modelPath': modelPath},
    );
    _isInitialized = result == true;
  }
  
  Future<String> generate({
    required String prompt,
    int maxTokens = 512,
    double temperature = 0.7,
  }) async {
    final result = await _channel.invokeMethod<String>(
      'generateText',
      {
        'prompt': prompt,
        'maxTokens': maxTokens,
        'temperature': temperature,
      },
    );
    return result ?? '[Error: No response]';
  }
  
  Future<void> dispose() async {
    await _channel.invokeMethod('disposeModel');
    _isInitialized = false;
  }
}
```

**Note:** Native implementation required in Swift (iOS) and Kotlin (Android). See ExecuTorch documentation for details.

---

# Part 2: Arny Mobile AI Architecture

## Architectural Strategies Overview

![Plan A vs Plan B Architecture](https://raw.githubusercontent.com/manumishra12/test/main/fig/plan_a_vs_plan_b.png)
*Figure 5: Architectural Comparison - Universal vs Specialized Hybrid*

### Plan A: The "Universal Monolith"

**Concept:** One large model that handles all tasks through system prompt conditioning.

**Characteristics:**
- Single model file (~2-3GB)
- Context switching via system prompt (instant)
- High RAM usage (constant ~2GB)
- Single fine-tuning run
- Must retrain entire model for updates

**Advantages:**
- Simple architecture
- No model loading delays
- Easier single codebase maintenance
- Faster development cycle

**Disadvantages:**
- High RAM usage (may crash on older devices)
- Catastrophic forgetting risk
- Larger storage requirement
- Difficult to A/B test individual capabilities

---

### Plan B: The "Specialized Hybrid" (Recommended)

**Concept:** Multiple small expert models coordinated by a lightweight gateway router.

**Characteristics:**
- Multiple model files (gateway + specialists)
- Context switching via model loading (2-4s GGUF, <1s PTE)
- Low RAM usage (one expert at a time)
- Separate fine-tuning runs per specialist
- Independent model updates

**Advantages:**
- Low RAM usage (safe for older devices)
- Modular updates
- Progressive app updates
- Easy A/B testing per capability
- Scales to many specialized skills

**Disadvantages:**
- Model loading delay when switching
- More complex orchestration
- More training runs required
- Larger total storage if all models downloaded

---

## Implementation Matrix

| Feature | Plan A GGUF | Plan A PTE | Plan B GGUF | Plan B PTE |
|---------|-------------|------------|-------------|------------|
| **Processor** | CPU | NPU/GPU | CPU | NPU/GPU |
| **RAM Impact** | High (~2.2GB) | Medium (~1.8GB) | Low (~500MB) | Lowest (~400MB) |
| **Switching** | Instant | Instant | Slow (2-4s) | Fast (<1s) |
| **Battery** | High | Low | Medium | Very Low |
| **Complexity** | Low | High | Medium | Very High |
| **iOS Safety** | Risk | Good | Excellent | Excellent |

---

## Plan A: Universal Monolith Implementation

### Data Preparation

```python
# prepare_universal_dataset.py
from datasets import Dataset, concatenate_datasets

SYSTEM_PROMPTS = {
    "wellness": "[AGENT:wellness] You are an empathetic wellness counselor.",
    "math": "[AGENT:math] You are a precise mathematics tutor.",
    "logistics": "[AGENT:logistics] You are an efficient logistics planner.",
}

def prepare_universal_dataset():
    # Load datasets
    wellness_data = load_wellness_conversations()
    math_data = load_math_problems()
    logistics_data = load_logistics_scenarios()
    
    # Format with system prompts
    wellness_formatted = format_dataset(wellness_data, SYSTEM_PROMPTS["wellness"])
    math_formatted = format_dataset(math_data, SYSTEM_PROMPTS["math"])
    logistics_formatted = format_dataset(logistics_data, SYSTEM_PROMPTS["logistics"])
    
    # Balance and combine
    min_samples = min(len(wellness_formatted), len(math_formatted), len(logistics_formatted))
    universal_dataset = concatenate_datasets([
        wellness_formatted.select(range(min_samples)),
        math_formatted.select(range(min_samples)),
        logistics_formatted.select(range(min_samples)),
    ]).shuffle(seed=42)
    
    return universal_dataset
```

### Flutter Implementation (Plan A with GGUF)

```dart
// lib/services/universal_ai_service_gguf.dart
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class UniversalAiServiceGguf {
  LlamaProcessor? _processor;
  String _currentRole = "wellness";
  bool _isInitialized = false;
  
  static const Map<String, String> _systemPrompts = {
    "wellness": "[AGENT:wellness] You are an empathetic wellness counselor.",
    "math": "[AGENT:math] You are a precise mathematics tutor.",
    "logistics": "[AGENT:logistics] You are an efficient logistics planner.",
  };
  
  // Initialize once - model stays in memory
  Future<void> initialize() async {
    if (_isInitialized) return;
    
    _processor = LlamaProcessor(
      'assets/models/arny_universal.gguf',
      nThreads: 4,
      nCtx: 2048,
      nBatch: 512,
    );
    await _processor!.initialize();
    _isInitialized = true;
    print('Universal model loaded: ~2GB RAM');
  }
  
  // Switch role instantly (just changes system prompt)
  void switchRole(String role) {
    if (_systemPrompts.containsKey(role)) {
      _currentRole = role;
      print('Switched to $role mode (instant, 0ms)');
    }
  }
  
  // Generate response with current role
  Stream<String> generate(String prompt) async* {
    if (!_isInitialized || _processor == null) {
      throw Exception('Model not initialized');
    }
    
    final fullPrompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
${_systemPrompts[_currentRole]}<|eot_id|><|start_header_id|>user<|end_header_id|>
$prompt<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''';
    
    await for (final token in _processor!.stream(
      fullPrompt,
      maxTokens: 512,
      temperature: 0.7,
      stop: ['<|eot_id|>'],
    )) {
      yield token;
    }
  }
  
  void dispose() {
    _processor?.dispose();
  }
}
```

**Usage Example:**
```dart
// In your Flutter app
final aiService = UniversalAiServiceGguf();

// Initialize once at app startup
await aiService.initialize();

// Switch between roles (instant)
aiService.switchRole('wellness');
await for (final token in aiService.generate('I feel stressed')) {
  print(token); // "It's natural to feel..."
}

// Switch to math (instant, no loading)
aiService.switchRole('math');
await for (final token in aiService.generate('Solve 2x + 5 = 15')) {
  print(token); // "Let's solve step by step..."
}
```

**Key Characteristics:**
- **RAM:** Constant ~2.2GB (model always loaded)
- **Switching:** Instant (0ms)
- **Best for:** High-end devices (8GB+ RAM)
- **Risk:** May crash on iPhone 12/13 with low memory

---

### Flutter Implementation (Plan A with PTE)

```dart
// lib/services/universal_ai_service_pte.dart
import 'package:flutter/services.dart';

class UniversalAiServicePte {
  static const MethodChannel _channel = 
      MethodChannel('com.example.app/executorch');
  
  String _currentRole = "wellness";
  bool _isInitialized = false;
  
  static const Map<String, String> _systemPrompts = {
    "wellness": "[AGENT:wellness] You are an empathetic wellness counselor.",
    "math": "[AGENT:math] You are a precise mathematics tutor.",
    "logistics": "[AGENT:logistics] You are an efficient logistics planner.",
  };
  
  // Initialize once - model loaded on native side
  Future<void> initialize() async {
    if (_isInitialized) return;
    
    final result = await _channel.invokeMethod<bool>(
      'initializeModel',
      {'modelPath': 'arny_universal.pte'},
    );
    
    if (result == true) {
      _isInitialized = true;
      print('Universal model loaded with NPU/GPU acceleration');
    }
  }
  
  // Switch role instantly (just changes prompt prefix)
  void switchRole(String role) {
    if (_systemPrompts.containsKey(role)) {
      _currentRole = role;
      print('Switched to $role mode (instant, 0ms)');
    }
  }
  
  // Generate response with hardware acceleration
  Future<String> generate(String prompt) async {
    if (!_isInitialized) {
      throw Exception('Model not initialized');
    }
    
    final fullPrompt = '''${_systemPrompts[_currentRole]}

User: $prompt
Assistant:''';
    
    final result = await _channel.invokeMethod<String>(
      'generateText',
      {
        'prompt': fullPrompt,
        'maxTokens': 512,
        'temperature': 0.7,
      },
    );
    
    return result ?? '[Error: No response]';
  }
  
  Future<void> dispose() async {
    await _channel.invokeMethod('disposeModel');
    _isInitialized = false;
  }
}
```

**Usage Example:**
```dart
// In your Flutter app
final aiService = UniversalAiServicePte();

// Initialize once at app startup
await aiService.initialize();

// Switch between roles (instant)
aiService.switchRole('wellness');
final response1 = await aiService.generate('I feel stressed');
print(response1); // Fast response with NPU/GPU

// Switch to math (instant, no loading)
aiService.switchRole('math');
final response2 = await aiService.generate('Solve 2x + 5 = 15');
print(response2); // Fast response with hardware acceleration
```

**Key Characteristics:**
- **RAM:** ~1.8GB (more efficient than GGUF)
- **Switching:** Instant (0ms)
- **Performance:** 2-3x faster than GGUF (NPU/GPU)
- **Complexity:** Requires native bridge implementation
- **Best for:** High-end devices with native dev expertise

---

## Plan B: Specialized Hybrid Implementation

### Gateway Model Training

```python
# train_gateway_model.py
from unsloth import FastLanguageModel
import json

gateway_training_data = [
    {"input": "I feel overwhelmed", 
     "output": json.dumps({"agent": "wellness", "confidence": 0.95})},
    {"input": "Solve: 3x + 7 = 22",
     "output": json.dumps({"agent": "math", "confidence": 0.98})},
    {"input": "Plan my route",
     "output": json.dumps({"agent": "logistics", "confidence": 0.92})},
]

# Use smaller model for gateway
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=512,
    load_in_4bit=True,
)

# Train and save
# trainer.train()
model.save_pretrained_gguf("gateway", tokenizer, quantization_method="q4_k_m")
```

### Flutter Orchestrator

```dart
// Hybrid AI Service (GGUF)
class HybridAiServiceGguf {
  late LlamaProcessor _gateway;
  LlamaProcessor? _currentSpecialist;
  String? _currentSpecialistName;
  
  Future<void> initialize() async {
    _gateway = LlamaProcessor('assets/models/gateway.gguf', nThreads: 2, nCtx: 512);
    await _gateway.initialize();
  }
  
  Future<RoutingDecision> classifyIntent(String input) async {
    final buffer = StringBuffer();
    await for (final token in _gateway.stream(input, maxTokens: 100)) {
      buffer.write(token);
      if (buffer.toString().contains('}')) break;
    }
    return RoutingDecision.fromJson(jsonDecode(buffer.toString()));
  }
  
  Future<void> _loadSpecialist(String name) async {
    if (_currentSpecialistName == name) return;
    
    _currentSpecialist?.dispose();
    _currentSpecialist = LlamaProcessor(
      'assets/models/$name.gguf',
      nThreads: 4,
      nCtx: 2048,
    );
    await _currentSpecialist!.initialize();
    _currentSpecialistName = name;
  }
  
  Stream<String> generate({required String prompt}) async* {
    final decision = await classifyIntent(prompt);
    
    yield '[Loading ${decision.agent} model...]';
    await _loadSpecialist(decision.agent);
    
    await for (final token in _currentSpecialist!.stream(prompt)) {
      yield token;
    }
  }
}

class RoutingDecision {
  final String agent;
  final double confidence;
  
  RoutingDecision({required this.agent, required this.confidence});
  
  factory RoutingDecision.fromJson(Map<String, dynamic> json) {
    return RoutingDecision(
      agent: json['agent'],
      confidence: json['confidence'],
    );
  }
}
```

**Usage Example:**
```dart
// In your Flutter app
final aiService = HybridAiServiceGguf();

// Initialize gateway (300MB)
await aiService.initialize();
print('Gateway loaded: ~300MB RAM');

// First wellness query (loads wellness model)
await for (final token in aiService.generate(prompt: 'I feel anxious')) {
  print(token); 
  // Output: "[Loading wellness model...]"
  // (2-4 seconds delay)
  // Then: "It's completely normal to feel..."
}

// Second wellness query (model already loaded, no delay)
await for (final token in aiService.generate(prompt: 'How to relax?')) {
  print(token); // Immediate response
}

// Switch to math (unloads wellness, loads math)
await for (final token in aiService.generate(prompt: 'Solve 3x + 7 = 22')) {
  print(token);
  // Output: "[Loading math model...]"
  // (2-4 seconds delay)
  // Then: "Let's solve step by step..."
}
```

**Key Characteristics:**
- **RAM:** Low ~500MB (only one specialist loaded)
- **Switching:** 2-4 seconds (model loading delay)
- **Best for:** All devices including budget phones
- **Safe:** Works reliably on iPhone 12+, Android 10+

---

### Flutter Implementation (Plan B with PTE)

```dart
// lib/services/hybrid_ai_service_pte.dart
import 'package:flutter/services.dart';
import 'dart:convert';

class HybridAiServicePte {
  static const MethodChannel _channel = 
      MethodChannel('com.example.app/executorch');
  
  bool _isInitialized = false;
  String? _currentSpecialist;
  
  static const Map<String, String> _modelPaths = {
    'gateway': 'gateway.pte',
    'wellness': 'wellness.pte',
    'math': 'math.pte',
    'logistics': 'logistics.pte',
  };
  
  // Initialize all models (memory-mapped, fast)
  Future<void> initialize() async {
    if (_isInitialized) return;
    
    // PTE models are memory-mapped, so we can "load" all without RAM penalty
    final result = await _channel.invokeMethod<bool>(
      'initializeModels',
      {'models': _modelPaths},
    );
    
    if (result == true) {
      _isInitialized = true;
      print('All models memory-mapped: ~400MB RAM total');
    }
  }
  
  // Classify intent using gateway
  Future<RoutingDecision> classifyIntent(String input) async {
    final result = await _channel.invokeMethod<String>(
      'classifyIntent',
      {'prompt': input},
    );
    
    final json = jsonDecode(result ?? '{}');
    return RoutingDecision(
      agent: json['agent'] as String,
      confidence: (json['confidence'] as num).toDouble(),
    );
  }
  
  // Switch specialist (fast with memory mapping)
  Future<void> _switchSpecialist(String name) async {
    if (_currentSpecialist == name) return;
    
    print('Switching to $name (<1s with memory mapping)');
    await _channel.invokeMethod('switchSpecialist', {'name': name});
    _currentSpecialist = name;
  }
  
  // Generate with appropriate specialist
  Future<String> generate(String prompt) async {
    if (!_isInitialized) {
      throw Exception('Models not initialized');
    }
    
    // Step 1: Classify intent
    final decision = await classifyIntent(prompt);
    print('Classified as: ${decision.agent} (${decision.confidence})');
    
    // Step 2: Switch to specialist (fast)
    await _switchSpecialist(decision.agent);
    
    // Step 3: Generate response
    final result = await _channel.invokeMethod<String>(
      'generateText',
      {
        'prompt': prompt,
        'maxTokens': 512,
        'temperature': 0.7,
      },
    );
    
    return result ?? '[Error: No response]';
  }
  
  Future<void> dispose() async {
    await _channel.invokeMethod('disposeModels');
    _isInitialized = false;
  }
}

class RoutingDecision {
  final String agent;
  final double confidence;
  
  RoutingDecision({required this.agent, required this.confidence});
}
```

**Usage Example:**
```dart
// In your Flutter app
final aiService = HybridAiServicePte();

// Initialize all models (memory-mapped)
await aiService.initialize();
print('All models ready: ~400MB RAM');

// First wellness query (switches to wellness)
final response1 = await aiService.generate('I feel anxious');
// Output: 
// "Classified as: wellness (0.95)"
// "Switching to wellness (<1s with memory mapping)"
// Response: "It's completely normal to feel..."

// Second wellness query (already on wellness, instant)
final response2 = await aiService.generate('How to relax?');
// No switching delay, immediate response

// Switch to math (fast switch <1s)
final response3 = await aiService.generate('Solve 3x + 7 = 22');
// Output:
// "Classified as: math (0.98)"
// "Switching to math (<1s with memory mapping)"
// Response: "Let's solve step by step..."
```

**Key Characteristics:**
- **RAM:** Lowest ~400MB (memory-mapped models)
- **Switching:** <1 second (fast with memory mapping)
- **Performance:** 2-3x faster than GGUF (NPU/GPU)
- **Complexity:** High (requires native implementation)
- **Best for:** Production apps needing best performance

---

## Implementation Comparison Summary

| Feature | Plan A GGUF | Plan A PTE | Plan B GGUF | Plan B PTE |
|---------|-------------|------------|-------------|------------|
| **Architecture** | Universal Monolith | Universal Monolith | Specialized Hybrid | Specialized Hybrid |
| **Model Count** | 1 (3B params) | 1 (3B params) | 4 (Gateway + 3 Specialists) | 4 (Gateway + 3 Specialists) |
| **Total Storage** | ~2GB | ~2GB | ~2GB (Gateway 300MB + Specialists 500MB each) | ~2GB |
| **RAM Usage** | Constant 2.2GB | Constant 1.8GB | Dynamic 500MB | Dynamic 400MB |
| **Context Switch** | Instant (0ms) | Instant (0ms) | 2-4 seconds | <1 second |
| **Inference Speed** | 0.5s per response | 0.3s per response | 0.5s per response | 0.3s per response |
| **Battery Impact** | High | Low | Medium | Very Low |
| **Dev Complexity** | Low (Pure Dart) | High (Native Code) | Medium (Pure Dart) | Very High (Native Code) |
| **Update Strategy** | Retrain entire model | Retrain entire model | Update individual specialists | Update individual specialists |
| **Device Safety** | Risk on old devices | Good | Excellent | Excellent |
| **Best Use Case** | Quick MVP on high-end | Performance MVP | Production-ready | Best performance |

**Quick Selection Guide:**

**Choose Plan A GGUF if:**
- You need fastest development (1 week)
- Testing MVP on high-end devices only
- Team has no native mobile expertise
- Instant context switching is critical

**Choose Plan A PTE if:**
- You need fast MVP with good performance
- Team has native mobile expertise
- Targeting premium devices
- Can invest 2-3 weeks for native setup

**Choose Plan B GGUF if:** (RECOMMENDED)
- Building production app
- Need wide device compatibility
- Want modular updates
- Team has no native mobile expertise
- Can accept 2-4s switching delay

**Choose Plan B PTE if:**
- Building premium production app
- Performance is critical (<1s everything)
- Team has native mobile expertise
- Budget allows 4-6 weeks native development

---

# Part 3: Performance Analysis & Recommendations

![Memory Usage Patterns](https://raw.githubusercontent.com/manumishra12/test/main/fig/memory_usage.png)
*Figure 6: Memory Usage Over Time - Plan A vs Plan B*

## Device Compatibility Matrix

| Device Category | RAM | Plan A GGUF | Plan A PTE | Plan B GGUF | Plan B PTE |
|-----------------|-----|-------------|------------|-------------|------------|
| **High-end** (iPhone 15 Pro, S24) | 8GB+ | Works | Works | Works | Best |
| **Mid-range** (iPhone 13, A54) | 4-6GB | Risk | Risk | Works | Recommended |
| **Budget** (iPhone SE 3, A34) | 4GB | Crashes | Crashes | Works | Works |
| **Old** (iPhone 11, S10) | 4GB | High Risk | High Risk | Slow | Acceptable |

## Response Time Comparison

| Scenario | Plan A GGUF | Plan A PTE | Plan B GGUF | Plan B PTE |
|----------|-------------|------------|-------------|------------|
| First message | 0.5s | 0.3s | 2-4s (load) | 1s (load) |
| Same context | 0.5s | 0.3s | 0.5s | 0.3s |
| Switch context | 0s | 0s | 2-4s (reload) | <1s (switch) |

## Final Recommendation

![Decision Tree](https://raw.githubusercontent.com/manumishra12/test/main/fig/decision_tree.png)
*Figure 7: Architecture Decision Tree - Choosing the Right Approach*

**Recommended: Plan B (Specialized Hybrid) with GGUF**

**Why This Combination?**

1. **Device Compatibility:** Ensures app stability across all devices (iPhone 12+, Android 10+)
2. **Development Efficiency:** Pure Dart implementation in 1-2 weeks
3. **Deployment Flexibility:** Progressive feature rollout with on-demand downloads
4. **Maintenance:** Independent model updates without full system retraining

**Migration Path to PTE:**
Once GGUF implementation is stable, migrate to PTE for performance optimization. The Plan B architecture remains unchanged - simply swap the inference engine.

**Justified when:**
- User feedback indicates performance concerns
- Analytics show conversation abandonment due to delays
- Team has native mobile development expertise
- Targeting premium devices with NPU capabilities

---


## Additional Resources

**Official Documentation:**
- Unsloth: https://github.com/unslothai/unsloth
- Llama.cpp: https://github.com/ggerganov/llama.cpp
- ExecuTorch: https://pytorch.org/executorch/
- llama_cpp_dart: https://pub.dev/packages/llama_cpp_dart

**Training Resources:**
- Unsloth Colab Notebooks: https://github.com/unslothai/unsloth/tree/main/notebooks
- Fine-tuning Guide: https://docs.unsloth.ai/
- Quantization Best Practices: https://huggingface.co/docs/optimum/
