#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright 2014 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
__author__ = 'api.nickm@gmail.com (Nick Mihailovski)'
import argparse
import sys
from googleapiclient.errors import HttpError
from googleapiclient import sample_tools
from oauth2client.client import AccessTokenRefreshError
import collections
import csv
import numpy as np
import pandas as pd 

def main(argv):
  # Authenticate and construct service.
  service, flags = sample_tools.init(
      argv, 'analytics', 'v3', __doc__, __file__,
      scope='https://www.googleapis.com/auth/analytics.readonly')
  # Try to make a request to the API. Print the results or handle errors.
  try:
    first_profile_id = get_first_profile_id(service)
    if not first_profile_id:
      print('Could not find a valid profile for this user.')
    else:
      results = get_parameters(service, first_profile_id)
      print_results(results)
  except TypeError as error:
    # Handle errors in constructing a query.
    print(('There was an error in constructing your query : %s' % error))
  except HttpError as error:
    # Handle API errors.
    print(('Arg, there was an API error : %s : %s' %
           (error.resp.status, error._get_reason())))
  except AccessTokenRefreshError:
    # Handle Auth errors.
    print ('The credentials have been revoked or expired, please re-run '
           'the application to re-authorize')


def get_first_profile_id(service):
  accounts = service.management().accounts().list().execute()
  if accounts.get('items'):
    firstAccountId = accounts.get('items')[0].get('id')
    webproperties = service.management().webproperties().list(
        accountId=firstAccountId).execute()
    if webproperties.get('items'):
      firstWebpropertyId = webproperties.get('items')[0].get('id')
      profiles = service.management().profiles().list(
          accountId=firstAccountId,
          webPropertyId=firstWebpropertyId).execute()
      if profiles.get('items'):
        return profiles.get('items')[0].get('id')
  return None

def get_parameters(service, profile_id):
  profile_id = "159135697"
  return service.data().realtime().get(
      ids='ga:' + profile_id,
      metrics='rt:totalEvents',
      dimensions='rt:eventAction,rt:eventLabel,rt:eventCategory', #'ga:source,ga:keyword',
      max_results='25').execute()

def print_results(results):
  print('Profile Name: %s' % results.get('profileInfo').get('profileName'))
  # Print header.
  output = []
  if results.get('rows', []):
    output = []
    matrix = [('%s' % row[0],'%s' % row[1].replace('[','').replace(']','').split(",")[0],'%s' % row[1].replace('[','').replace(']','').split(",")[1] ) for row in results.get('rows')]
    np.savetxt("../predictions/data/%s.csv" % row[1][1], matrix, fmt='%s', delimiter= ",")  
    print(matrix)  
  else:
    print('No Rows Found')

if __name__ == '__main__':
  main(sys.argv)


"""Simple intro to using the Google Analytics API v3.

This application demonstrates how to use the python client library to access
Google Analytics data. The sample traverses the Management API to obtain the
authorized user's first profile ID. Then the sample uses this ID to
contstruct a Core Reporting API query to return the metrics we selected.

Before you begin, you must sigup for a new project in the Google APIs console:
https://code.google.com/apis/console

Then register the project to use OAuth2.0 for installed applications.

Finally you will need to add the client id, client secret, and redirect URL
into the client_secrets.json file that is in the same directory as this sample."""
