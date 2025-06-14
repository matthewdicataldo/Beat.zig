# ZigPulse Integration Guide

## Overview

ZigPulse uses a modular architecture with a convenient bundle file for single-import usage. This document explains the integration options and why we moved away from traditional amalgamation.

## Current Status

### Modular Version (Recommended)
- **Status**: ✅ Stable and fully functional
- **Location**: `src/core.zig` (main entry point)
- **Usage**: Import via Zig's module system
- **Benefits**: 
  - Clean namespace separation
  - Better compile-time optimization
  - Easier debugging and development
  - No symbol conflicts

### Bundle File
- **Status**: ✅ Stable
- **Location**: `zigpulse.zig` (in root directory)
- **Purpose**: Single import convenience while preserving modular structure
- **Requirements**: The `src/` directory must be present

## Why Amalgamation is Challenging in Zig

Unlike C, where amalgamation (like SQLite's single-file distribution) works well, Zig presents unique challenges:

1. **Strong Type System**: Zig's compile-time type checking prevents ambiguous symbols
2. **Module System**: Zig has explicit imports and namespaces, not preprocessor includes
3. **No Macros**: Can't use text substitution to resolve conflicts
4. **Comptime Evaluation**: Makes runtime string manipulation insufficient

## Best Practices

### For Library Users

**Recommended Approach**: Use the modular version
```zig
// In your build.zig
const zigpulse_module = b.addModule("zigpulse", .{
    .root_source_file = .{ .path = "libs/ZigPulse/src/core.zig" },
});
exe.root_module.addImport("zigpulse", zigpulse_module);
```

**Alternative**: Copy the entire `src/` directory
- Preserves module structure
- Allows local modifications
- No amalgamation issues

### For Library Developers

If you need to create a working amalgamation:

1. **Rename Conflicting Symbols**:
   - `lockfree.Task` → `lockfree.TaskWrapper`
   - Module-level constants need unique prefixes

2. **Use Proper Namespacing**:
   ```zig
   pub const zigpulse = struct {
       pub const lockfree = struct { ... };
       pub const topology = struct { ... };
       // ... other modules
   };
   ```

3. **Consider Alternatives**:
   - Zig package manager (when available)
   - Git submodules
   - Build-time dependency management

## Technical Details

The current amalgamation script (`scripts/amalgamate.zig`) attempts to:
1. Read all imported modules recursively
2. Inline their contents
3. Remove duplicate imports
4. Create a single file

However, it fails because:
- Multiple modules define symbols with the same name
- The flattening process loses namespace boundaries
- Cross-module references become ambiguous

## Future Plans

1. **Short Term**: Document the modular approach as the primary integration method
2. **Medium Term**: Create a proper AST-based amalgamation tool
3. **Long Term**: Wait for Zig package manager maturity

## Conclusion

While single-file distribution is convenient, it conflicts with Zig's design philosophy. The modular approach is more idiomatic and reliable for Zig projects.

For now, use:
- **Modular import** for production use
- **Single-file** only for experimentation or if you're willing to manually fix conflicts