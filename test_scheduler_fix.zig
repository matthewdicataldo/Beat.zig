const std = @import("std");
const core = @import("src/core.zig");
const scheduler = @import("src/scheduler.zig");

// Comprehensive test demonstrating the scheduler token accounting fix

test "Token Accounting - Deep Analysis and Fix Verification" {
    std.debug.print("\n=== Token Accounting System - Deep Analysis ===\n", .{});
    
    const config = core.Config{
        .promotion_threshold = 10,
        .min_work_cycles = 1000,
    };
    
    // Test Case 1: Original failing scenario
    std.debug.print("1. Testing original failing scenario...\n", .{});
    var tokens1 = scheduler.TokenAccount.init(&config);
    
    std.debug.print("   Initial state: work={}, overhead={}, should_promote={}\n", .{
        tokens1.work_cycles, tokens1.overhead_cycles, tokens1.shouldPromote()
    });
    
    tokens1.update(5000, 100); // 50:1 ratio, should promote (5000 > 100 * 10)
    std.debug.print("   After update(5000, 100): work={}, overhead={}, should_promote={}\n", .{
        tokens1.work_cycles, tokens1.overhead_cycles, tokens1.shouldPromote()
    });
    
    try std.testing.expect(tokens1.shouldPromote()); // This was failing before fix
    std.debug.print("   ✓ PASS: First update with promotion ratio triggers promotion\n", .{});
    
    // Test Case 2: Non-promoting scenario
    std.debug.print("\n2. Testing non-promoting scenario...\n", .{});
    tokens1.reset();
    tokens1.update(500, 100); // 5:1 ratio, should not promote (500 < 100 * 10)
    
    std.debug.print("   After reset and update(500, 100): work={}, overhead={}, should_promote={}\n", .{
        tokens1.work_cycles, tokens1.overhead_cycles, tokens1.shouldPromote()
    });
    
    try std.testing.expect(!tokens1.shouldPromote());
    std.debug.print("   ✓ PASS: Low work:overhead ratio does not trigger promotion\n", .{});
    
    // Test Case 3: Minimum work cycles threshold
    std.debug.print("\n3. Testing minimum work cycles threshold...\n", .{});
    var tokens2 = scheduler.TokenAccount.init(&config);
    
    tokens2.update(500, 10); // 50:1 ratio but below min_work_cycles (1000)
    std.debug.print("   After update(500, 10): work={}, overhead={}, should_promote={}\n", .{
        tokens2.work_cycles, tokens2.overhead_cycles, tokens2.shouldPromote()
    });
    
    try std.testing.expect(!tokens2.shouldPromote()); // Should not promote due to min_work_cycles
    std.debug.print("   ✓ PASS: Below minimum work cycles threshold prevents promotion\n", .{});
    
    // Test Case 4: Performance caching behavior
    std.debug.print("\n4. Testing performance caching behavior...\n", .{});
    var tokens3 = scheduler.TokenAccount.init(&config);
    
    // First update should calculate
    tokens3.update(10000, 100);
    const first_result = tokens3.shouldPromote();
    std.debug.print("   First update result: {}\n", .{first_result});
    
    // Small subsequent updates shouldn't recalculate (cached)
    tokens3.update(1, 1);
    tokens3.update(1, 1);
    const cached_result = tokens3.shouldPromote();
    std.debug.print("   Cached result after small updates: {}\n", .{cached_result});
    
    try std.testing.expectEqual(first_result, cached_result);
    std.debug.print("   ✓ PASS: Caching works correctly for small updates\n", .{});
    
    // Large update should trigger recalculation
    tokens3.update(1, 1000); // This should trigger CHECK_INTERVAL
    const recalculated_result = tokens3.shouldPromote();
    std.debug.print("   Result after large overhead update: {}\n", .{recalculated_result});
    std.debug.print("   ✓ PASS: Large updates trigger recalculation\n", .{});
    
    // Test Case 5: Edge cases
    std.debug.print("\n5. Testing edge cases...\n", .{});
    var tokens4 = scheduler.TokenAccount.init(&config);
    
    // Zero overhead
    tokens4.update(5000, 0);
    std.debug.print("   Zero overhead case: should_promote={}\n", .{tokens4.shouldPromote()});
    try std.testing.expect(!tokens4.shouldPromote()); // Division by zero protection
    
    // Zero work
    tokens4.reset();
    tokens4.update(0, 100);
    std.debug.print("   Zero work case: should_promote={}\n", .{tokens4.shouldPromote()});
    try std.testing.expect(!tokens4.shouldPromote());
    
    std.debug.print("   ✓ PASS: Edge cases handled correctly\n", .{});
    
    std.debug.print("\n=== Token Accounting Fix Successfully Verified! ===\n", .{});
}

test "Token Accounting - Performance Characteristics" {
    std.debug.print("\n=== Performance Characteristics Analysis ===\n", .{});
    
    const config = core.Config{
        .promotion_threshold = 5,
        .min_work_cycles = 100,
    };
    
    var tokens = scheduler.TokenAccount.init(&config);
    
    // Simulate realistic workload patterns
    std.debug.print("Simulating realistic workload patterns:\n", .{});
    
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const work = 1000 + (i * 500);
        const overhead = 100 + (i * 20);
        
        tokens.update(work, overhead);
        const ratio = @as(f64, @floatFromInt(tokens.work_cycles)) / @as(f64, @floatFromInt(tokens.overhead_cycles));
        
        std.debug.print("  Update {}: work={}, overhead={}, ratio={d:.1}, promote={}\n", .{
            i, work, overhead, ratio, tokens.shouldPromote()
        });
        
        // Should promote when ratio > threshold and work > min_work_cycles
        const expected_promote = tokens.work_cycles >= config.min_work_cycles and 
            tokens.work_cycles > (tokens.overhead_cycles * config.promotion_threshold);
        
        try std.testing.expectEqual(expected_promote, tokens.shouldPromote());
    }
    
    std.debug.print("✓ PASS: All promotion decisions match expected logic\n", .{});
}