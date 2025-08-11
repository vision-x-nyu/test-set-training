# TsT LLM Integration Tasks

This directory contains detailed task documentation for integrating LLM-based bias detection into the TsT evaluation framework.

## Overview

The implementation is organized into 4 phases, each building on the previous:

1. **Phase 1**: Minimal refactor to create base abstractions
2. **Phase 2**: Production-ready LLM system with multi-GPU support  
3. **Phase 3**: Unified evaluation framework for all model types
4. **Phase 4**: Move benchmark-specific logic to benchmark modules

## Task Documents

### Main Planning
- [`implementation-roadmap.md`](./implementation-roadmap.md) - Complete overview and timeline

### Phase-Specific Tasks
- [`phase-1-minimal-refactor.md`](./phase-1-minimal-refactor.md) - Base abstractions (2-3 days)
- [`phase-2-llm-system.md`](./phase-2-llm-system.md) - Production LLM system (1-2 weeks)
- [`phase-3-unified-evaluation.md`](./phase-3-unified-evaluation.md) - Unified framework (3-4 days) 
- [`phase-4-benchmark-integration.md`](./phase-4-benchmark-integration.md) - Benchmark integration (1 week)

## Quick Start

To begin implementation:

1. **Read the roadmap**: Start with `implementation-roadmap.md` for context
2. **Begin Phase 1**: Follow `phase-1-minimal-refactor.md` step by step
3. **Set up environment**: Ensure Ray, vLLM, and LlamaFactory are working
4. **Create feature branch**: Work incrementally with proper version control

## Key Design Principles

- **Backward Compatibility**: Existing RF functionality continues working
- **Clean Abstractions**: Separate models, evaluation, and data formatting
- **Production Ready**: Multi-GPU support, memory management, type safety
- **DataEnvGym Patterns**: Follow proven architectural patterns

## Success Criteria

- **Performance**: 3-5x speedup from multi-GPU inference
- **Memory**: <50MB storage per k-fold experiment  
- **Code Quality**: 90%+ test coverage, zero lint warnings
- **Usability**: Same interface for RF and LLM modes

Each phase has detailed success criteria and testing strategies in its respective document.

## Dependencies

New dependencies to add during implementation:

```bash
# Core dependencies
uv add ray[default]  # Multi-GPU parallel inference
uv add pydantic     # Type-safe data models
uv add typing-extensions  # Enhanced typing support

# Optional but recommended  
uv add loguru       # Better logging
uv add ulid-py      # Unique identifiers
```

## Estimated Timeline

- **Total**: 5-6 weeks
- **Phase 1**: 2-3 days (minimal refactor)
- **Phase 2**: 1-2 weeks (LLM system)
- **Phase 3**: 3-4 days (unified evaluation)  
- **Phase 4**: 1 week (benchmark integration)

## Questions?

Each task document is designed to be self-contained with enough detail to implement without additional context. If you have questions about specific phases:

1. Check the relevant phase document first
2. Refer to the roadmap for overall context
3. Look at the DataEnvGym reference implementation for patterns

The design prioritizes incremental progress - each phase provides value independently while building toward the complete solution.
