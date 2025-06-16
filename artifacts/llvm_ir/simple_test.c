// Simple test file for Souper superoptimization validation
// Contains patterns that Souper should detect and optimize

int bitwiseTest(int x) {
    // Pattern: redundant operations that should be optimized
    x = x + 0;      // Should be optimized away
    x = x * 1;      // Should be optimized away  
    x = x | 0;      // Should be optimized away
    x = x & 0xFFFFFFFF; // Should be optimized away for 32-bit int
    
    // Pattern: double negation
    if (!(!(x > 0))) {
        x = x ^ x;  // Always 0, should be optimized
    }
    
    // Pattern: bitwise clear lowest set bit
    x = x & (x - 1);
    
    return x;
}

int arithmeticTest(int a, int b) {
    // Pattern: strength reduction opportunities
    int result = a * 8; // Should become a << 3
    result = result / 4; // Should become result >> 2
    
    // Pattern: algebraic identities
    result = result + b - b; // Should be optimized to just result
    result = result * 2 / 2; // Should be optimized to just result
    
    return result;
}

int main() {
    int test1 = bitwiseTest(42);
    int test2 = arithmeticTest(10, 5);
    return test1 + test2;
}