$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = if ($args.Count -gt 0) { [int]$args[0] } else { 5173 }
$prefix = "http://localhost:$port/"

$types = @{
  ".html" = "text/html; charset=utf-8"
  ".css" = "text/css; charset=utf-8"
  ".js" = "text/javascript; charset=utf-8"
  ".json" = "application/json; charset=utf-8"
  ".png" = "image/png"
  ".jpg" = "image/jpeg"
  ".jpeg" = "image/jpeg"
  ".svg" = "image/svg+xml"
  ".wav" = "audio/wav"
}

$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add($prefix)
$listener.Start()
Write-Host "Sanjjang server running at $prefix"

try {
  while ($listener.IsListening) {
    $context = $listener.GetContext()
    $path = [Uri]::UnescapeDataString($context.Request.Url.AbsolutePath.TrimStart("/"))
    if ([string]::IsNullOrWhiteSpace($path)) {
      $path = "index.html"
    }

    $candidate = Join-Path $root $path
    $fullPath = [System.IO.Path]::GetFullPath($candidate)
    $rootPath = [System.IO.Path]::GetFullPath($root)

    if (-not $fullPath.StartsWith($rootPath) -or -not (Test-Path $fullPath -PathType Leaf)) {
      $context.Response.StatusCode = 404
      $bytes = [System.Text.Encoding]::UTF8.GetBytes("Not found")
    } else {
      $extension = [System.IO.Path]::GetExtension($fullPath).ToLowerInvariant()
      $context.Response.ContentType = if ($types.ContainsKey($extension)) { $types[$extension] } else { "application/octet-stream" }
      $bytes = [System.IO.File]::ReadAllBytes($fullPath)
    }

    $context.Response.ContentLength64 = $bytes.Length
    $context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    $context.Response.OutputStream.Close()
  }
} finally {
  $listener.Stop()
}
