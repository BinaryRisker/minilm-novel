/* JS 侧矩阵乘法基准：复刻 minilm.js 里的生产实现（i-k-j 缓存友好顺序） */
var REAL = Float32Array;

function mm(A, B, M, K, N, C) {
  for (var i = 0; i < M; i++) {
    var ai = i * K, ci = i * N;
    for (var j = 0; j < N; j++) C[ci + j] = 0;
    for (var k = 0; k < K; k++) {
      var a = A[ai + k];
      if (a === 0) continue;
      var bk = k * N;
      for (var j2 = 0; j2 < N; j2++) C[ci + j2] += a * B[bk + j2];
    }
  }
}

function rand(n) {
  var a = new REAL(n);
  for (var i = 0; i < n; i++) a[i] = Math.random() * 2 - 1;
  return a;
}

function bench(M, K, N, iters) {
  var A = rand(M * K), B = rand(K * N), C = new REAL(M * N);
  mm(A, B, M, K, N, C);  // 预热

  var t0 = Date.now();
  for (var it = 0; it < iters; it++) mm(A, B, M, K, N, C);
  var dt = (Date.now() - t0) / 1000;

  var flops = 2 * M * K * N * iters;
  var gf = flops / dt / 1e9;
  console.log('JS  ' + M + 'x' + K + ' @ ' + K + 'x' + N + '  x' + iters +
              '  →  ' + dt.toFixed(2) + 's  |  ' + gf.toFixed(2) + ' GFLOPS');
  return gf;
}

console.log('--- JS 手写矩阵乘法（Node V8 / Intel N100 4 核单线程）---');
var a = bench(256, 192, 768, 300);    // 对应 FFN 层
var b = bench(1024, 1024, 1024, 30);  // 大矩阵，BLAS 最擅长的场景
console.log('平均: ' + ((a + b) / 2).toFixed(2) + ' GFLOPS\n');
