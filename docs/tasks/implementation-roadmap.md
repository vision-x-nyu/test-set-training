# TsT LLM Integration - Implementation Roadmap

## Overview

This document outlines the complete plan for integrating LLM-based bias detection into the TsT (Train-on-Test-Set) evaluation framework. The approach is designed as a 4-phase implementation that progressively improves the architecture while maintaining backward compatibility.

## Current State

The TsT framework currently supports Random Forest-based bias detection with feature engineering. We have a prototype LLM implementation in `llm_utils.py` that works but lacks production-readiness. Key issues:

- **Code Duplication**: 80% overlap between RF and LLM evaluation loops
- **RF-Centric Design**: Core abstractions assume feature-based models
- **Hardcoded Logic**: Benchmark-specific conversions scattered throughout
- **Scalability Issues**: Single-GPU inference, no proper memory management

## Design Principles

1. **Backward Compatibility**: Existing RF functionality must continue working
2. **Clean Abstractions**: Separate concerns (models, evaluation, data formatting)
3. **Production Ready**: Multi-GPU support, proper memory management, type safety
4. **Extensible**: Easy to add new model types or benchmarks
5. **DataEnvGym Patterns**: Follow proven patterns from the reference implementation

## Phase-by-Phase Breakdown

### Phase 1: Minimal Refactor (2-3 days)
**Goal**: Create base abstractions for clean LLM integration

**Key Components**:
- `BiasModel` protocol for generic model interface
- `ModelEvaluator` abstract base for evaluation strategies  
- Extract common cross-validation logic
- Maintain RF compatibility through inheritance

**Success Criteria**:
- No code duplication between RF and future LLM evaluators
- Clean extension point for LLM implementation
- Existing RF tests pass unchanged

**Files Created/Modified**:
- `src/TsT/core/protocols.py` (new)
- `src/TsT/core/evaluators.py` (new)
- `src/TsT/core/cross_validation.py` (new)
- `src/TsT/evaluation.py` (major refactor)

### Phase 2: Clean LLM System (1-2 weeks)
**Goal**: Production-ready LLM system following DataEnvGym patterns

**Key Components**:
- Multi-GPU Ray-based parallel inference
- Efficient LoRA training with LlamaFactory
- Proper GPU memory management
- Type-safe data models with Pydantic
- Composition pattern (predictor + trainer = trainable predictor)

**Success Criteria**:
- 3-5x speedup from multi-GPU inference
- Memory-efficient k-fold training (5 adapters < 50MB total)
- Clean separation of predictor/trainer concerns
- Proper error handling and resource cleanup

**Package Structure**:
```
src/TsT/llm/
├── predictors/     # vLLM single/multi-GPU
├── trainers/       # LlamaFactory wrapper
├── trainable/      # Composed trainable predictor
├── data/          # Pydantic models
└── utils/         # I/O and config utilities
```

### Phase 3: Unified Evaluation (3-4 days)
**Goal**: Single evaluation framework for all model types

**Key Components**:
- `UnifiedCrossValidator` handles all CV logic
- Rich `EvaluationResult` objects with fold-level metadata
- Consistent progress tracking and reporting
- Post-processing hooks for model-specific metadata

**Success Criteria**:
- Single source of truth for cross-validation
- Identical result format across model types
- Easy to add new model types
- Detailed result objects for analysis

**Benefits**:
- Eliminates remaining code duplication
- Consistent user experience
- Better testing and maintenance
- Rich metadata for analysis

### Phase 4: Benchmark Integration (1 week)
**Goal**: Move all benchmark-specific logic to benchmark modules

**Key Components**:
- `TextDataFormatter` protocol for benchmark-specific formatting
- Generic conversion functions using formatter interface
- Base formatters for common patterns (MC, numerical)
- Benchmark-specific formatters in benchmark modules

**Success Criteria**:
- Core evaluation code has zero benchmark knowledge
- New benchmarks can be added without touching core code
- Consistent text formatting interface
- Maintainable benchmark-specific logic

**Architecture**:
```
Core Framework (generic)
    ↕ TextDataFormatter protocol
Benchmark Modules (specific)
    ├── video_mme/text_formatter.py
    ├── vsi/text_formatter.py
    └── mmmu/text_formatter.py
```

## Timeline and Dependencies

```
Week 1: Phase 1 (Minimal Refactor)
├── Day 1-2: Create base abstractions
├── Day 2-3: Extract CV logic and update evaluation
└── Day 3: Testing and validation

Week 2-3: Phase 2 (LLM System)  
├── Week 2: Core infrastructure (predictors, trainers, data models)
├── Week 3: Ray integration, memory management, testing
└── Integration with Phase 1 abstractions

Week 4: Phase 3 (Unified Evaluation)
├── Day 1-2: UnifiedCrossValidator and result objects
├── Day 3-4: Update evaluators and main evaluation function
└── Testing and documentation

Week 5: Phase 4 (Benchmark Integration)
├── Day 1-2: Create formatter protocols and base classes
├── Day 3-4: Migrate existing benchmarks
└── Day 5: Testing and cleanup
```

## Risk Mitigation

### Technical Risks
1. **Ray Setup Complexity**: Start with single-GPU, add Ray incrementally
2. **GPU Memory Issues**: Implement thorough testing and monitoring
3. **LlamaFactory Integration**: Use proven patterns from DataEnvGym
4. **Performance Regressions**: Maintain comprehensive benchmarks

### Schedule Risks
1. **Scope Creep**: Stick to defined phases, document future improvements
2. **Testing Overhead**: Write tests incrementally, not at the end
3. **Integration Issues**: Test integrations early and often

## Success Metrics

### Performance
- **Inference Speed**: 3-5x improvement with multi-GPU
- **Memory Usage**: < 50MB storage per k-fold experiment
- **Training Time**: < 3 minutes per fold for small models

### Code Quality
- **Test Coverage**: > 90% for new code
- **Unit Tests**: All critical functions have pytest coverage
- **Linting**: Zero warnings/errors
- **Type Safety**: Full mypy compliance
- **Regression Tests**: Existing functionality continues working

### Usability
- **Backward Compatibility**: All existing RF tests pass
- **API Consistency**: Same interface for RF and LLM modes
- **Documentation**: Complete examples and migration guides

## Testing Strategy

Each phase should include comprehensive testing:

### Unit Tests (pytest)
- **Critical Functions**: CV logic, evaluators, data conversion
- **Protocol Compliance**: Ensure models implement interfaces correctly
- **Error Handling**: Test edge cases and error conditions
- **Mock Dependencies**: Test components in isolation

### Integration Tests
- **End-to-End**: Full evaluation pipelines with real data
- **Compatibility**: Old and new approaches produce identical results
- **Memory Management**: Proper cleanup in LLM components

### Regression Tests
- **Benchmark Results**: Ensure scores remain consistent
- **Performance**: Monitor evaluation speed and memory usage
- **API Stability**: Existing code continues working

## Post-Implementation

### Immediate Benefits
1. **Production Ready**: LLM evaluation ready for paper experiments
2. **Scalable**: Multi-GPU support for large-scale evaluation
3. **Maintainable**: Clean architecture reduces technical debt
4. **Extensible**: Framework ready for future model types

### Future Opportunities
1. **More Model Types**: Transformer features, gradient-based attribution
2. **Advanced Training**: Multi-task learning, meta-learning approaches
3. **Distributed Evaluation**: Ray clusters for massive benchmarks
4. **Interactive Analysis**: Real-time evaluation dashboards

## Getting Started

To begin implementation:

1. **Read Phase 1 Document**: `docs/tasks/phase-1-minimal-refactor.md`
2. **Set Up Development Environment**: Ensure Ray, vLLM, LlamaFactory work
3. **Create Feature Branch**: Start with minimal refactor
4. **Run Existing Tests**: Establish baseline before changes

The implementation is designed to be done incrementally, with each phase providing value independently while building toward the complete solution.

## Questions and Decisions

### Resolved
- ✅ Use composition over inheritance for trainable predictors
- ✅ Follow DataEnvGym patterns for proven reliability
- ✅ Implement Ray-based parallelism for scalability
- ✅ Phase approach to manage complexity and risk

### Open Questions
- How many GPUs will be available for testing?
- Should we implement distributed training across nodes?
- What level of configurability do we need for LLM training?
- Do we need support for custom evaluation metrics?

These questions can be addressed during implementation as requirements become clearer.
