# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Docker support with Miniconda base image
- Docker Compose configuration with GPU support options
- Updated README with installation instructions for both Docker and manual setup
- Pre-built PyTorch Geometric extensions installation to avoid compilation issues

### Changed
- Updated Dockerfile to use conda for dependency management instead of pip
- Added volume for conda package cache to speed up subsequent builds
- Improved documentation for setup and installation

### Fixed
- Resolved compatibility issues with PyTorch Geometric extensions by using pre-built wheels
