#!/usr/bin/env python
import rospy
import dynamic_reconfigure.client
import yaml


def main():
    import sys

    if len(sys.argv) < 3:
        print 'load_dynamic_config <node_name> <file.yaml>'
        exit()

    node_name = sys.argv[1]
    filename = sys.argv[2]
    rospy.init_node('load_dynamic_config',
                    anonymous=True,
                    disable_signals=True)
    client = dynamic_reconfigure.client.Client(node_name)
    node_config = client.get_configuration()

    f = open(filename)
    rospy.loginfo('Using config from ' + str(filename))
    new_config = yaml.load(f)
    client.update_configuration(new_config)

    try:
        while True:
            rospy.sleep(3.0)
    finally:
        rospy.loginfo('Spin broken! restoring your old configuration.')
        response = client.update_configuration(node_config)
        rospy.signal_shutdown("interrupted")
