param(
    [string]$EnvironmentPath = 'C:\Users\Admin\.venvs\adbench-smoke-py310',
    [switch]$ForceRecreate
)

$ErrorActionPreference = 'Stop'

$expectedPython = '3.10.11'
$resolvedEnvironmentPath = [System.IO.Path]::GetFullPath($EnvironmentPath)

$pythonVersion = (& py -3.10 -c "import platform; print(platform.python_version())").Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne $expectedPython) {
    throw "Python $expectedPython is required; py -3.10 resolved to '$pythonVersion'."
}

if (Test-Path -LiteralPath $resolvedEnvironmentPath) {
    if (-not $ForceRecreate) {
        throw "Environment already exists at '$resolvedEnvironmentPath'. Re-run with -ForceRecreate to replace it."
    }

    $allowedRoot = [System.IO.Path]::GetFullPath((Join-Path $env:USERPROFILE '.venvs'))
    if (-not $resolvedEnvironmentPath.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove an environment outside '$allowedRoot'."
    }

    Remove-Item -LiteralPath $resolvedEnvironmentPath -Recurse -Force
}

$parent = Split-Path -Parent $resolvedEnvironmentPath
New-Item -ItemType Directory -Force -Path $parent | Out-Null
& py -3.10 -m venv $resolvedEnvironmentPath

$python = Join-Path $resolvedEnvironmentPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual-environment creation failed: '$python' was not created."
}

& $python -m pip install --upgrade 'pip==23.0.1' 'setuptools==83.0.0' 'wheel==0.47.0'

$packages = @(
    'absl-py==2.5.0',
    'astunparse==1.6.3',
    'certifi==2026.6.17',
    'cffi==2.1.0',
    'charset-normalizer==3.4.9',
    'colorama==0.4.6',
    'combo==0.1.3',
    'contourpy==1.3.2',
    'copulas==0.14.1',
    'cryptography==49.0.0',
    'cycler==0.12.1',
    'filelock==3.29.7',
    'flatbuffers==25.12.19',
    'fonttools==4.63.0',
    'fsspec==2026.6.0',
    'gast==0.7.0',
    'google-auth==2.55.2',
    'google-auth-oauthlib==1.4.0',
    'google-pasta==0.2.0',
    'grpcio==1.82.1',
    'h5py==3.14.0',
    'idna==3.18',
    'Jinja2==3.1.6',
    'joblib==1.5.3',
    'keras==2.15.0',
    'kiwisolver==1.5.0',
    'libclang==18.1.1',
    'llvmlite==0.48.0',
    'Markdown==3.10.2',
    'markdown-it-py==4.2.0',
    'MarkupSafe==3.0.3',
    'matplotlib==3.10.9',
    'mdurl==0.1.2',
    'ml-dtypes==0.3.2',
    'mpmath==1.3.0',
    'namex==0.1.0',
    'narwhals==2.23.0',
    'networkx==3.4.2',
    'numba==0.66.0',
    'numpy==1.26.4',
    'oauthlib==3.3.1',
    'opt_einsum==3.4.0',
    'optree==0.19.1',
    'packaging==26.2',
    'pandas==2.3.3',
    'patsy==1.0.2',
    'pillow==12.3.0',
    'plotly==6.9.0',
    'protobuf==4.25.9',
    'pyasn1==0.6.4',
    'pyasn1_modules==0.4.2',
    'pycparser==3.0',
    'Pygments==2.20.0',
    'pyod==1.0.0',
    'pyparsing==3.3.2',
    'python-dateutil==2.9.0.post0',
    'pytz==2026.2',
    'requests==2.34.2',
    'requests-oauthlib==2.0.0',
    'rich==15.0.0',
    'scikit-learn==1.0.2',
    'scipy==1.15.3',
    'six==1.17.0',
    'statsmodels==0.14.6',
    'sympy==1.14.0',
    'tensorboard==2.15.2',
    'tensorboard-data-server==0.7.2',
    'tensorflow-cpu==2.15.1',
    'tensorflow-estimator==2.15.0',
    'tensorflow-intel==2.15.1',
    'tensorflow-io-gcs-filesystem==0.31.0',
    'termcolor==3.3.0',
    'threadpoolctl==3.6.0',
    'torch==2.2.2',
    'tqdm==4.68.4',
    'typing_extensions==4.16.0',
    'tzdata==2026.3',
    'urllib3==2.7.0',
    'Werkzeug==3.1.8',
    'wget==3.2',
    'wrapt==1.14.2'
)

& $python -m pip install $packages
& $python -m pip check

$installedPython = (& $python -c "import platform; print(platform.python_version())").Trim()
if ($installedPython -ne $expectedPython) {
    throw "Created environment reports Python '$installedPython', expected '$expectedPython'."
}

& $python -c "import numpy, scipy, pandas, sklearn, pyod"
& $python -c "import adbench.myutils; import adbench.datasets.data_generator"
& $python -c "import adbench.baseline.PyOD; import adbench.baseline.Customized.run"
& $python -c "from adbench.run import RunPipeline"

Write-Host "Verified ADBench environment created at: $resolvedEnvironmentPath"
Write-Host "Activate with: & '$resolvedEnvironmentPath\Scripts\Activate.ps1'"
