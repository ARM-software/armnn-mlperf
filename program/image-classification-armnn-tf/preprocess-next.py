#
# Copyright (c) 2019 dividiti Ltd.
# Copyright (c) 2019 Arm Ltd.
#
# SPDX-License-Identifier: MIT.
#

import os

def ck_preprocess(i):
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
  def dep_install_env(dep, var): return i['deps'][dep]['dict']['customize']['install_env'].get(var)
  
  MODEL_FILE = ''
  MODEL_DIR = dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_ROOT')
  for filename in os.listdir(MODEL_DIR):
    if filename.endswith('.pb'):
      MODEL_FILE = filename
      MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
  if not MODEL_FILE:
    return {'return': 1, 'error': 'Tf graph is not found in the selected model package'}

  # Setup parameters for program
  new_env = {}
  files_to_push_by_path = {}

  if i['target_os_dict'].get('remote','') == 'yes':
    # For Android we need only filename without full path  
    new_env['RUN_OPT_GRAPH_FILE'] = MODEL_FILE
    files_to_push_by_path['RUN_OPT_GRAPH_PATH'] = MODEL_PATH
  else:
    new_env['RUN_OPT_GRAPH_FILE'] = MODEL_PATH

  new_env['RUN_OPT_DATA_LAYOUT'] = dep_install_env('weights', 'ML_MODEL_DATA_LAYOUT')

  print('--------------------------------\n')
  return {
    'return': 0,
    'new_env': new_env,
    'files_to_push_by_path': files_to_push_by_path,
  }
