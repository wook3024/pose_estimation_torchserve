# import pathlib
# from setuptools import setup, find_packages

# # The directory containing this file
# HERE = pathlib.Path(__file__).parent

# # The text of the README file
# README = (HERE / "README.md").read_text()

# # This call to setup() does all the work
# setup(
#     name="shinuk",
#     version="1.0.5",
#     description="pose_estimation_module for torchserve",
#     long_description=README,
#     long_description_content_type="text/markdown",
#     url="https://github.com/wook3024/pose_estimation_torchserve/shinuk",
#     author="shinuk",
#     author_email="wook3024@gmail.com",
#     license="MIT",
#     zip_safe=False,
#     classifiers=[
#         "License :: OSI Approved :: MIT License",
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3.7",
#     ],
#     packages=find_packages(),
#     include_package_data=True,
#     install_requires=["numpy"],
#     # entry_points={
#     #     "console_scripts": [
#     #         "realpython=reader.__main__:main",
#     #     ]
#     # },
# )


from setuptools import setup, find_packages

setup(
    name             = 'shinuk',
    version          = '1.1.0',
    description      = 'Python wrapper for pose_estimation',
    author           = 'Shinuk Yi',
    author_email     = 'wook3024@gmail.com',
    url              = 'https://github.com/wook3024/pose_estimation_torchserve/shinuk',
    download_url     = 'https://github.com/wook3024/pose_estimation_torchserve/shinuk',
    install_requires = [ ],
    packages         = find_packages(),
    keywords         = ['pose_estimation', 'torchserve'],
    python_requires  = '>=3',
    classifiers      = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)