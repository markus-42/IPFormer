

import os
import pytorch_sphinx_theme
import subprocess
import sys
from m2r import MdInclude
from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath('..'))


project = 'MMDetection3D'
copyright = '2020-2023, OpenMMLab'
author = 'MMDetection3D Authors'

version_file = '../mmdet3d/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


release = get_version()


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
]

autodoc_mock_imports = [
    'matplotlib', 'nuscenes', 'PIL', 'pycocotools', 'pyquaternion',
    'terminaltables', 'mmdet3d.version', 'mmdet3d.ops'
]
autosectionlabel_prefix_document = True

templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmdetection3d'
        },
        {
            'name':
            '算法库',
            'children': [
                {
                    'name': 'MMAction2',
                    'url': 'https://github.com/open-mmlab/mmaction2',
                },
                {
                    'name': 'MMClassification',
                    'url': 'https://github.com/open-mmlab/mmclassification',
                },
                {
                    'name': 'MMCV',
                    'url': 'https://github.com/open-mmlab/mmmmcv',
                },
                {
                    'name': 'MMDetection',
                    'url': 'https://github.com/open-mmlab/mmdetection',
                },
                {
                    'name': 'MMEditing',
                    'url': 'https://github.com/open-mmlab/mmediting',
                },
                {
                    'name': 'MMGeneration',
                    'url': 'https://github.com/open-mmlab/mmgeneration',
                },
                {
                    'name': 'MMOCR',
                    'url': 'https://github.com/open-mmlab/mmocr',
                },
                {
                    'name': 'MMPose',
                    'url': 'https://github.com/open-mmlab/mmpose',
                },
                {
                    'name': 'MMSegmentation',
                    'url': 'https://github.com/open-mmlab/mmsegmentation',
                },
                {
                    'name': 'MMTracking',
                    'url': 'https://github.com/open-mmlab/mmtracking',
                },
            ]
        },
        {
            'name':
            'OpenMMLab',
            'children': [
                {
                    'name': '官网',
                    'url': 'https://openmmlab.com/'
                },
                {
                    'name': 'GitHub',
                    'url': 'https://github.com/open-mmlab/'
                },
                {
                    'name': '推特',
                    'url': 'https://twitter.com/OpenMMLab'
                },
                {
                    'name': '知乎',
                    'url': 'https://zhihu.com/people/openmmlab'
                },
            ]
        },
    ]
}

html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

latex_documents = [
    (master_doc, 'mmcv.tex', 'mmcv Documentation', 'MMCV Contributors',
     'manual'),
]

copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True


def builder_inited_handler(app):
    subprocess.run(['./stat.py'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
    app.add_config_value('no_underscore_emphasis', False, 'env')
    app.add_config_value('m2r_parse_relative_links', False, 'env')
    app.add_config_value('m2r_anonymous_references', False, 'env')
    app.add_config_value('m2r_disable_inline_math', False, 'env')
    app.add_directive('mdinclude', MdInclude)
    app.add_config_value('recommonmark_config', {
        'auto_toc_tree_section': 'Contents',
        'enable_eval_rst': True,
    }, True)
    app.add_transform(AutoStructify)
