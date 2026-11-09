package internal

import (
	"time"

	"github.com/codecrafters-io/tester-utils/tester_definition"
)

var testerDefinition = tester_definition.TesterDefinition{
	AntiCheatTestCases: []tester_definition.TestCase{},
	ExecutableFileName: "your_program.sh",
	TestCases: []tester_definition.TestCase{
		{Slug: "zf8", TestFunc: testReadSafetensors},
		{Slug: "pq6", TestFunc: testEmbeddings},
		{Slug: "mo0", TestFunc: testAttention},
		{Slug: "hu4", TestFunc: testTransformerBlock},
		{Slug: "wi3", TestFunc: testLogits},
		{Slug: "hh8", TestFunc: testGreedyDecode},
		{Slug: "df2", TestFunc: testSampling},
		{Slug: "rq5", TestFunc: testKVCache},
		// Up to 90 seconds to download GPT-2's weights, which usually takes
		// about 15, then up to 300 seconds for the learner's program. The 30
		// seconds left over cover the tester's own work, and the total stays
		// well under the platform's 10-minute limit, so the learner sees the
		// tester's message rather than a cutoff.
		{Slug: "zr7", TestFunc: testRunGPT2, Timeout: 7 * time.Minute},
	},
}
