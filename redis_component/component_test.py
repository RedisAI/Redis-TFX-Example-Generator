# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for redis_component.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from redis_component import component
from redis_proto import redis_config_pb2
from redis_proto import redis_hash_query_pb2
import tensorflow as tf

from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
from google.protobuf import json_format


class ComponentTest(tf.test.TestCase):

    def setUp(self):
        super(ComponentTest, self).setUp()
        self.conn_config = redis_config_pb2.RedisConnConfig(
            host='localhost', port=6379)

    def _extract_conn_config(self, custom_config):
        unpacked_custom_config = example_gen_pb2.CustomConfig()
        json_format.Parse(custom_config, unpacked_custom_config)

        conn_config = redis_config_pb2.RedisConnConfig()
        unpacked_custom_config.custom_config.Unpack(conn_config)
        return conn_config

    def testConstruct(self):
        redis_example_gen = component.RedisExampleGen(
            self.conn_config, query=redis_hash_query_pb2.RedisHashQuery(hash_key_pattern='*', schema=[
                {
                    'name': 'pickup_community_area',
                    'type': 'integer'
                },
                {
                    'name': 'fare',
                    'type': 'float'
                },
                {
                    'name': 'trip_start_month',
                    'type': 'integer'
                },
                {
                    'name': 'trip_start_hour',
                    'type': 'integer'
                },
                {
                    'name': 'trip_start_day',
                    'type': 'integer'
                },
                {
                    'name': 'trip_start_timestamp',
                    'type': 'integer'
                },
                {
                    'name': 'pickup_latitude',
                    'type': 'float'
                },
                {
                    'name': 'pickup_longitude',
                    'type': 'float'
                },
                {
                    'name': 'dropoff_latitude',
                    'type': 'float'
                },
                {
                    'name': 'dropoff_longitude',
                    'type': 'float'
                },
                {
                    'name': 'trip_miles',
                    'type': 'float'
                },
                {
                    'name': 'pickup_census_tract',
                    'type': 'integer'
                },
                {
                    'name':'dropoff_census_tract',
                    'type':'float'
                },
                {
                    'name': 'payment_type',
                    'type': 'string'
                },
                {
                    'name':'company',
                    'type':'string'
                },
                {
                    'name':'trip_seconds',
                    'type':'float'
                },
                {
                    'name':'dropoff_community_area',
                    'type':'float'
                },
                {
                    'name':'tips',
                    'type':'float'
                },
                {
                    'name':'big_tipper',
                    'type':'integer'
                }
               ]))
        self.assertEqual(
            self.conn_config,
            self._extract_conn_config(
                redis_example_gen.exec_properties['custom_config']))
        self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                         redis_example_gen.outputs['examples'].type_name)
        artifact_collection = redis_example_gen.outputs['examples'].get()
        self.assertEqual(1, len(artifact_collection))

    def testBadConstruction(self):
        empty_config = redis_config_pb2.RedisConnConfig()
        self.assertRaises(
            RuntimeError,
            component.RedisExampleGen,
            conn_config=empty_config,
            query=None,
            input_config=None)


if __name__ == '__main__':
    tf.test.main()
